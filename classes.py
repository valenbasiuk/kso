

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from pathlib import Path
import shutil
def distance_sq(pos1, pos2): #special distance
    x=(pos1.x-pos2.x)*1.25
    y=pos1.y-pos2.y
    return x**2+y**2

class Point:
    __slots__ = ['x', 'y']
    def __getstate__(self):
        return (self.x, self.y)
    def __setstate__(self, state):
        self.x, self.y = state
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __add__(self,other):
        return Point(self.x+other.x,self.y+other.y)
    def __sub__(self,other):
        return Point(self.x-other.x,self.y-other.y)
    #scale
    def __mul__(self,other):
        if isinstance(other,Point):
            return self.x*other.x+ self.y*other.y
        return Point(self.x*other,self.y*other)
    def __truediv__(self,other):
        return Point(self.x/other,self.y/other)
    def __neg__(self):
        return Point(-self.x,-self.y)
    def __eq__(self,other):
        return self.x==other.x and self.y==other.y
    def dist(self,other):
        return math.sqrt(self.sq_dist(other))
    def sq_dist(self, other):
        return (self.x-other.x)**2+(self.y-other.y)**2
    def cross(self, other):
        return self.x*other.y-self.y*other.x

    def __abs__(self):
        return math.sqrt(self.x**2+self.y**2)

    def __repr__(self):
        return f"{round(self.x,1)} {round(self.y,1)}"
    def __call__(self):
        return (self.x,self.y)
    def __hash__(self):
        return hash((self.x, self.y))
    def copy(self):
        return Point(self.x,self.y)
    

class Key:
    def __init__(self, lx, uy, width=1, offset=0):
        self._pos=Point(lx,uy)
        self._width=width
        self._offset=offset
        self.fpos=self._pos+Point(self._width/2+self._offset,0.5)

def check_config_file(name):
    a=Path(f"config/{name}")

    return a.exists()

def check_required_config_file(name):
    assert check_config_file(name), f"\aREQUIRED FILE {name.upper()} IN CONFIG FOLDER IN ORDER TO RUN THIS SCRIPT!"

class Layout:
    def __init__(self):
        self._init_layout()
        self._init_finger()
        self._init_parameters()
        self._init_keys()
        self._init_keystrokes()
        self._init_artist()

        self._init_assigned_keys()
        self._init_home_keys()
        self._init_max_finger_dists()
        self._precompute()
        self._init_visual()
          
    def _init_layout(self):
        self.KEY_WIDTH={}

        check_required_config_file("key_widths.json")

        try:
            with open('config/key_widths.json','r') as file:
                self.KEY_WIDTH=json.load(file)
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN key_widths.json FILE: {e}")
            raise SystemExit

        self.keys={}
        cur_y=0
        with open('config/layout.txt','r') as file:
            for line in file:
                cur_x=0
                for key in line.split():
                    self._is_key_correct_type(key, 'layout.txt')
                    if key in self.KEY_WIDTH:
                        is_list=False
                        has_offset=False
                        if isinstance(self.KEY_WIDTH[key],list):
                            is_list=True
                            has_offset=len(self.KEY_WIDTH[key])>1
                        
                        self.keys[key]=Key(cur_x,cur_y,self.KEY_WIDTH[key][0] if is_list else self.KEY_WIDTH[key], self.KEY_WIDTH[key][1] if is_list and has_offset else 0)

                        cur_x+=self.KEY_WIDTH[key][0]
                    else:
                        self.keys[key]=Key(cur_x,cur_y)
                        cur_x+=1
                cur_y+=1
        self.num_rows=cur_y
  
    def _init_keys(self):
        self._init_shift_layer()
        self._init_alt_layer()
        self._init_shift_alt_layer()
        self._init_fixed_keys()
        self._init_available_keys()   
        self.key_positions = [self.keys[key].fpos for key in self.idx2key] 

    def _init_shift_layer(self):
        if not check_config_file("available_shift_keys.txt"):
            shutil.copyfile("config/available_keys.txt", "config/available_shift_keys.txt")
        if not check_config_file("fixed_shift_keys.json"):
            with open("config/fixed_shift_keys.json", "w") as file:
                json.dump({}, file) 

    def _init_alt_layer(self):
        if not check_config_file("available_alt_keys.txt"):
            with open("config/available_alt_keys.txt", "w") as file:
                file.write("")
        if not check_config_file("fixed_alt_keys.json"):
            with open("config/fixed_alt_keys.json", "w") as file:
                json.dump({}, file)

    def _init_shift_alt_layer(self):
        if not check_config_file("available_shift_alt_keys.txt"):
            with open("config/available_shift_alt_keys.txt", "w") as file:
                file.write("")
        if not check_config_file("fixed_shift_alt_keys.json"):
            with open("config/fixed_shift_alt_keys.json", "w") as file:
                json.dump({}, file)

    def _init_fixed_keys(self):
        chat_matters = True
        try:
            with open('config/parameters.json', 'r') as file:
                paras = json.load(file)
                chat_matters = paras.get("chat_matters", True)
        except Exception:
            pass
        self.chat_matters = chat_matters

        file_name = 'fixed_keys.json'
        check_required_config_file(file_name)
        self.fixed_keys = []
        try:
            with open(f'config/{file_name}', 'r', encoding='utf-8') as file:
                self.fixed_keys.append(json.load(file))
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
            raise SystemExit

        shift_variances = {'sft', 'lsft', 'shift', 'rsft', 'lshift', 'rshift'}
        alt_variances = {'alt', 'lalt', 'ralt'}
        special_variances = {
            'bs': {'bspc', 'backspace', 'bs', '<'},
            'sp': {'sp', 'space', 'spc', '_'}
        }
        
        has_shift = False
        self.shift = 'lsft'
        self.shift_name = 'lsft'
        self.alt = None
        self.alt_name = 'alt'
        self.chat = 't' if self.chat_matters else None
        self.chat_name = 'chat'
        self.home = None
        self.home_name = 'home'
        self.specials = {
            'bs': [None, None, None],
            'sp': [None, None, None]
        }

        for key, remap in self.fixed_keys[0].items():
            self._is_key_correct_type(key, file_name)
            self._does_key_exist(key, self.keys, file_name, 'layout.txt')

            if self.chat_matters and ((isinstance(remap, str) and remap == self.chat_name) or (isinstance(remap, list) and (self.chat_name == remap[0] or (len(remap) > 1 and self.chat_name == remap[1])))):
                self.chat = key
            elif isinstance(remap, str):
                if remap in shift_variances:
                    self.shift = key
                    self.shift_name = remap
                    has_shift = True
                elif remap in alt_variances:
                    self.alt = key
                    self.alt_name = remap
                elif remap == 'home':
                    self.home = key
                    self.home_name = remap
                else:
                    for k, v in special_variances.items():
                        if remap in v:
                            self.specials[k][0] = key
                            self.specials[k][1] = remap
            elif not isinstance(remap, str):
                raise TypeError(f"\nKEY SLOT [{key.upper()}] NEEDS TO ASSOCIATED WITH A KEY TYPE STRING, NOT [{remap}] ({type(remap)})!")

        if not has_shift and self.shift not in self.keys:
            raise ValueError(f"\nPLEASE MAP [SHIFT] KEY WITH A KEYSLOT IN {file_name} FILE FIRST!")

        if self.chat_matters:
            self._does_key_exist(self.chat, self.keys, file_name, 'layout.txt')
            if self.chat in self.fixed_keys[0]:
                remap = self.fixed_keys[0][self.chat]
                if (isinstance(remap, str) and remap == self.chat_name) or (isinstance(remap, list) and len(remap) == 1 and self.chat_name == remap[0]):
                    self.fixed_keys[0].pop(self.chat)
                elif isinstance(remap, list) and len(remap) > 1:
                    self.fixed_keys[0][self.chat] = self.fixed_keys[0][self.chat][1 if remap[0] == self.chat_name else 0]
        else:
            if self.chat is not None and self.chat in self.fixed_keys[0]:
                remap = self.fixed_keys[0][self.chat]
                if (isinstance(remap, str) and remap == self.chat_name) or (isinstance(remap, list) and len(remap) == 1 and self.chat_name == remap[0]):
                    self.fixed_keys[0].pop(self.chat)
                elif isinstance(remap, list) and len(remap) > 1:
                    self.fixed_keys[0][self.chat] = self.fixed_keys[0][self.chat][1 if remap[0] == self.chat_name else 0]

        if not has_shift:
            self.fixed_keys[0][self.shift] = 'lsft'
        if self.alt is not None:
            self.fixed_keys[0][self.alt] = self.alt_name

        # Load fixed shift keys
        file_name = 'fixed_shift_keys.json'
        try:
            with open(f'config/{file_name}', 'r', encoding='utf-8') as file:
                self.fixed_keys.append(json.load(file))
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
            raise SystemExit

        for key, remap in self.fixed_keys[1].items():
            self._is_key_correct_type(key, file_name)
            self._does_key_exist(key, self.keys, file_name, 'layout.txt')
            if not isinstance(remap, str):
                raise TypeError(f"\nKEY SLOT [{key.upper()}] IN {file_name} FILE NEEDS TO ASSOCIATED WITH A KEY TYPE STRING, NOT [{remap}] ({type(remap)})!")

        # Load fixed alt keys
        file_name = 'fixed_alt_keys.json'
        try:
            with open(f'config/{file_name}', 'r', encoding='utf-8') as file:
                self.fixed_keys.append(json.load(file))
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
            raise SystemExit

        for key, remap in self.fixed_keys[2].items():
            self._is_key_correct_type(key, file_name)
            self._does_key_exist(key, self.keys, file_name, 'layout.txt')
            if not isinstance(remap, str):
                raise TypeError(f"\nKEY SLOT [{key.upper()}] IN {file_name} FILE NEEDS TO ASSOCIATED WITH A KEY TYPE STRING, NOT [{remap}] ({type(remap)})!")

        # Load fixed shift+alt keys
        file_name = 'fixed_shift_alt_keys.json'
        try:
            with open(f'config/{file_name}', 'r', encoding='utf-8') as file:
                self.fixed_keys.append(json.load(file))
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
            raise SystemExit

        for key, remap in self.fixed_keys[3].items():
            self._is_key_correct_type(key, file_name)
            self._does_key_exist(key, self.keys, file_name, 'layout.txt')
            if not isinstance(remap, str):
                raise TypeError(f"\nKEY SLOT [{key.upper()}] IN {file_name} FILE NEEDS TO ASSOCIATED WITH A KEY TYPE STRING, NOT [{remap}] ({type(remap)})!")

    def _init_available_keys(self):
        file_name = 'available_keys.txt'
        check_required_config_file(file_name)
        self.remaps = [{key: remap for key, remap in layer.items()} for layer in self.fixed_keys]
        with open(f'config/{file_name}', 'r') as file:
            for line in file:
                for key in line.split():
                    self._is_key_correct_type(key, file_name)
                    self._does_key_exist(key, self.keys, file_name, 'layout.txt')
                    if key in self.fixed_keys[0] and (not self.chat_matters or key != self.chat):
                        continue
                    self.remaps[0][key] = key

        # Load available shift keys
        file_name = 'available_shift_keys.txt'
        check_required_config_file(file_name)
        with open(f'config/{file_name}', 'r') as file:
            for line in file:
                for key in line.split():
                    self._is_key_correct_type(key, file_name)
                    self._does_key_exist(key, self.keys, file_name, 'layout.txt')
                    if key in self.fixed_keys[1]:
                        continue
                    if key == self.shift or key == self.home or any(key == self.specials[k][0] for k in self.specials):
                        continue
                    self.remaps[1][key] = key

        # Load available alt keys
        file_name = 'available_alt_keys.txt'
        check_required_config_file(file_name)
        with open(f'config/{file_name}', 'r') as file:
            for line in file:
                for key in line.split():
                    self._is_key_correct_type(key, file_name)
                    self._does_key_exist(key, self.keys, file_name, 'layout.txt')
                    if key in self.fixed_keys[2]:
                        continue
                    if key == self.shift or key == self.alt or key == self.home or any(key == self.specials[k][0] for k in self.specials):
                        continue
                    self.remaps[2][key] = key

        # Load available shift+alt keys
        file_name = 'available_shift_alt_keys.txt'
        check_required_config_file(file_name)
        with open(f'config/{file_name}', 'r') as file:
            for line in file:
                for key in line.split():
                    self._is_key_correct_type(key, file_name)
                    self._does_key_exist(key, self.keys, file_name, 'layout.txt')
                    if key in self.fixed_keys[3]:
                        continue
                    if key == self.shift or key == self.alt or key == self.home or any(key == self.specials[k][0] for k in self.specials):
                        continue
                    self.remaps[3][key] = key

        if self.chat_matters:
            assert self.remaps[0].get(self.chat) is not None, "BUG: Chat key is not present in remaps[0]!"

        self.key2idx = [{}, {}, {}, {}]
        self.sizes = [len(layer) for layer in self.remaps]
        
        if (self.sizes[2] > 0 or self.sizes[3] > 0) and self.alt is None:
            raise ValueError("\nPLEASE MAP [ALT] KEY WITH A KEYSLOT IN fixed_keys.json FILE FIRST AS YOU HAVE ACTIVE KEYS ON ALT / SHIFT+ALT LAYER!")

        self.idx2key = []
        j = 0
        for i, layer in enumerate(self.remaps):
            for key in sorted(layer, key=lambda k: (self.keys[k]._pos.y, self.keys[k]._pos.x)):
                self.key2idx[i][key] = j
                self.idx2key.append(key)
                j += 1

        self.layered_available_keys = [[j for key, j in layer.items() if key not in self.fixed_keys[i]] for i, layer in enumerate(self.key2idx)]
        self.available_keys = [key_idx for layer in self.layered_available_keys for key_idx in layer]

        self.counterparts = {}
        for i, layer in enumerate(self.key2idx):
            for key, key_idx in layer.items():
                cps = []
                for other_i, other_layer in enumerate(self.key2idx):
                    if other_i != i and key in other_layer:
                        cps.append(other_layer[key])
                self.counterparts[key_idx] = cps

        self.fixed_keys = {self.key2idx[i][key]: remap for i, layer in enumerate(self.fixed_keys) for key, remap in layer.items()}

        self.chat_i = [self.key2idx[i][self.chat] for i, layer in enumerate(self.key2idx) if self.chat is not None and self.chat in self.key2idx[i]]
        self.shift_i = [self.key2idx[i][self.shift] for i, layer in enumerate(self.key2idx) if self.shift in self.key2idx[i]]
        self.alt_i = [self.key2idx[i][self.alt] for i, layer in enumerate(self.key2idx) if self.alt is not None and self.alt in self.key2idx[i]] if self.alt is not None else None
        self.home_i = [self.key2idx[i][self.home] for i, layer in enumerate(self.key2idx) if self.home in self.key2idx[i]] if self.home is not None else None
        # print(self.fixed_keys)
        # print(self.key2idx)
        # print(self.idx2key)
    
    def _init_finger(self):
        self.FINGER_CODE={'pinky':0,'ring':1,'middle':2,'index':3,'thumb':4} #DO NOT TOUCH
        self.HAND_CODE={'left':0,'right':1} #DO NOT TOUCH

        self.idx2finger=[f"{hand}_{finger}" for hand, hand_idx in sorted(self.HAND_CODE.items(),key=lambda x: x[1]) for finger, finger_idx in sorted(self.FINGER_CODE.items(),key=lambda x: x[1])]
        self.finger2idx={finger: idx for idx, finger in enumerate(self.idx2finger)}

    def _init_keystrokes(self):
        self.keystrokes=[]
        self.total_weights=0
        file_name="keystrokes.json"
        check_required_config_file(file_name)
        missing_w_count=0

        #special keys(eat the whole key and don't have shift counterparts)
        self.special_keybinds=set()

        try:
            with open(f'config/{file_name}','r', encoding='utf-8') as file:
                keystrokes_dict=json.load(file)
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
            raise SystemExit
        for name, keystroke in keystrokes_dict.items():
            if isinstance(keystroke,list):
                keystrokes_dict[name]={"keys":keystroke,}
                keystroke=keystrokes_dict[name]
            else:
                if "keys" not in keystroke:
                    raise ValueError(f"\nPLEASE ENTER KEYS FOR [{name.upper()}] IN {file_name} FILE FIRST!")
            
            # Standardize keys to options
            if isinstance(keystroke["keys"], list) and len(keystroke["keys"]) > 0 and isinstance(keystroke["keys"][0], list):
                keystroke["options"] = keystroke["keys"]
            else:
                keystroke["options"] = [keystroke["keys"]]

            if "weight" in keystroke:
                self.total_weights += keystroke["weight"]
            else:
                missing_w_count += 1

            shift_safe = keystroke.get("shift_safe", False)
            if shift_safe:
                for opt in keystroke["options"]:
                    for key in opt:
                        self.special_keybinds.add(key)

        average_w = self.total_weights / (len(keystrokes_dict) - missing_w_count) if (len(keystrokes_dict) - missing_w_count) > 0 else 1.0
        self.total_weights += missing_w_count * average_w
        self.keystrokes = []
        for name, keystroke in keystrokes_dict.items():
            if "weight" not in keystroke:
                print(f"WARNING: Keystroke [{name}] in keystrokes.json file doesn't have a weight! Setting it to average value.")
                keystroke["weight"] = average_w

            touch_shift = keystroke.get("touch_shift", False) or keystroke.get("shift_safe", False)
            opts = keystroke["options"]
            # touch_shift=True means: these keys live on the shift layer.
            # The keystroke sequence is used as-is; normalize_keystrokes will
            # automatically insert shift when it sees a shift-layer keybind.
            # (No longer prepends lsft/home to the sequence.)

            self.keystrokes.append({
                "options": opts,
                "weight": keystroke["weight"],
                "touch_shift": touch_shift
            })

        for i in range(len(self.keystrokes)):
            self.keystrokes[i]["weight"] /= self.total_weights

        mod_set = {self.shift_name}
        if self.chat_matters:
            mod_set.add(self.chat_name)
        if not self.skip_home:
            mod_set.add(self.home_name)
        if self.alt is not None:
            mod_set.add(self.alt_name)
        self.keybinds = sorted({key for data in self.keystrokes for opt in data['options'] for key in opt}.union(mod_set))
        self.keybind2idx = {keybind: i for i, keybind in enumerate(self.keybinds)}

        self.available_keybinds = [i for i, key in enumerate(self.keybinds) if key not in self.fixed_keys.values() and key != self.chat_name]

        #encode for home, shift, backspace, alt
        self.shift_kbi=self.keybind2idx[self.shift_name]
        self.home_kbi=self.keybind2idx[self.home_name] if not self.skip_home else None
        if self.alt is not None:
            self.alt_kbi=self.keybind2idx[self.alt_name]
        else:
            self.alt_kbi=None

        for k in self.specials.keys():
            if k == 'bs' and self.skip_backspace:
                continue
            if self.specials[k][1] in self.keybind2idx:
                self.specials[k][2]=self.keybind2idx[self.specials[k][1]]
        
        #special keys
        self.special_keybinds=[self.keybind2idx[key] for key in self.special_keybinds]
        self.available_skb=[kb_idx for kb_idx in self.special_keybinds if kb_idx not in self.fixed_keys.values()]
        
        self.no_shift_variance_skb_set=set(self.special_keybinds)
        self.no_shift_variance_skb_set.discard(self.shift_kbi) #pop shift
        if self.alt_kbi is not None:
            self.no_shift_variance_skb_set.discard(self.alt_kbi)

        self.no_alt_variance_skb_set=set(self.special_keybinds)
        self.no_alt_variance_skb_set.discard(self.shift_kbi)
        if self.alt_kbi is not None:
            self.no_alt_variance_skb_set.discard(self.alt_kbi)

        if not self.skip_home:
            self.special_keybinds.append(self.home_kbi)
            if self.home_i is None:
                self.available_skb.append(self.home_kbi)
        for k in self.specials:
            if k == 'bs' and self.skip_backspace:
                continue
            if self.specials[k][2] is not None:
                kb=self.specials[k][2]
                self.special_keybinds.append(kb)
                self.no_shift_variance_skb_set.add(kb)
                self.no_alt_variance_skb_set.add(kb)
                if self.specials[k][0] is None:
                    self.available_skb.append(kb)
        #special set
        self.special_kb_set=set(self.special_keybinds)
        self.available_skb_set=set(self.available_skb)


        #frequency of keys
        freq=[0]*len(self.keybinds)
        self.total_keybinds=0
        for data in self.keystrokes:
            opt_weight = data['weight'] / len(data['options'])
            for opt in data['options']:
                for key in opt:
                    if key==self.shift_name or (self.alt is not None and key==self.alt_name): #skip modifiers
                        continue
                    freq[self.keybind2idx[key]]+=opt_weight
                    self.total_keybinds+=opt_weight

        #Chat probability
        if self.chat_matters:
            freq[self.keybind2idx[self.chat_name]]+=average_w/self.total_weights*len(self.keystrokes)
            self.total_keybinds+=average_w/self.total_weights*len(self.keystrokes)
        
        #probability of keys in keystrokes
        self.key_probs=tuple(f/self.total_keybinds for f in freq)


        #turn dict to list and encode into indices
        self.keystrokes=[{"options": [[self.keybind2idx[key] for key in opt] for opt in data["options"]], "weight": data["weight"], "touch_shift": data["touch_shift"]} for data in self.keystrokes]

        # When chat is ignored, remove the 'chat' keybind from all keystroke option
        # sequences so that options like ['chat','i','m'] become ['i','m'] and the
        # evaluator can still pick a valid option (instead of skipping all of them
        # because chat has no key slot, leading to 999999 → NaN scores).
        if not self.chat_matters and self.chat_name in self.keybind2idx:
            chat_kbi = self.keybind2idx[self.chat_name]
            filtered_ks = []
            for ks_data in self.keystrokes:
                new_opts = [
                    [k for k in opt if k != chat_kbi]
                    for opt in ks_data["options"]
                ]
                new_opts = [opt for opt in new_opts if opt]  # drop empty options
                if new_opts:
                    filtered_ks.append({"options": new_opts, "weight": ks_data["weight"], "touch_shift": ks_data["touch_shift"]})
            if filtered_ks:
                self.keystrokes = filtered_ks


        #encode fixed keys into keybind indices (if it is in self.keybinds) 
        #keyIdx: keybind --> keyIdx: keybindIdx
        encoded_fixed_keys={}
        for keyIdx, remap in self.fixed_keys.items():
            if remap in self.keybind2idx:
                encoded_fixed_keys[keyIdx]=self.keybind2idx[remap]
            else:
                pass
                # raise ValueError(f"KEYBIND {remap} IN fixed_keys.json FILE IS UNUSED!")
                print(f"WARNING: Keybind {remap} in fixed_keys.json file is unused, skipping it")
        self.fixed_keys=encoded_fixed_keys
        # self.fixed_keys={keyIdx:(self.keybind2idx[remap] if remap in self.keybind2idx else remap) for keyIdx, remap in self.fixed_keys.items()}


    def _init_assigned_keys(self):
        self.assigned_keys={}
        file_name='assigned_fingers.json'
        check_required_config_file(file_name)
        try:
            with open(f'config/{file_name}','r') as file:
                self.assigned_keys=json.load(file)
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
            raise SystemExit

        # key_idx2finger_idx  – single finger (or None when shared / unassigned)
        # key_idx2finger_candidates – ALL candidate fingers for each key slot;
        #   empty list  → key has no assigned finger (e.g. number row)
        #   1-elem list → unambiguous single finger
        #   N-elem list → shared key; evaluator picks closest at runtime
        self.key_idx2finger_idx=[None]*len(self.idx2key)
        self.key_idx2finger_candidates=[[] for _ in range(len(self.idx2key))]

        for finger, keys in self.assigned_keys.items():
            self._validate_finger(finger,file_name)
            finger_idx=self.finger2idx[finger]
            for key in keys:
                for i,layer in enumerate(self.key2idx):
                    if key not in layer:
                        continue
                    slot=self.key2idx[i][key]
                    if finger_idx not in self.key_idx2finger_candidates[slot]:
                        self.key_idx2finger_candidates[slot].append(finger_idx)

        # Populate key_idx2finger_idx: only set when exactly one candidate
        for slot, candidates in enumerate(self.key_idx2finger_candidates):
            if len(candidates) == 1:
                self.key_idx2finger_idx[slot] = candidates[0]
            elif len(candidates) > 1:
                # Shared key – warn once per (finger, key) pair for first duplicate
                key_name = self.idx2key[slot]
                for finger_idx in candidates[1:]:
                    finger_name = self.idx2finger[finger_idx]
                    print(f"WARNING: Finger [{finger_name.upper()}] is pressing the same key [{key_name}] as another finger in {file_name}! Skipping it.")

    def _init_artist(self):
        self.fig, self.ax = plt.subplots(figsize=(6.4,3.6))
        bgc='#13161B'
        self.fig.patch.set_facecolor(bgc)
        self.ax.set_facecolor(bgc)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.invert_yaxis()
        # Hide all axis decorations (ticks, labels, frame, title)
        self.ax.axis('off')

    # ── Display helpers ──────────────────────────────────────────────────────
    def _display_label(self, char):
        """Map internal keybind names to short display labels."""
        label_map = {'home': 'HM', 'bspc': 'BS', 'spc': 'SP', 'chat': 'CH'}
        return label_map.get(char, char)

    def _init_visual(self):
        self.rects=[]
        self.texts=[]
        ec='#3F72AF'
        fc='#00ADB5'
        tc='#E3FDFD'
        stc='#FFD580'       # shift layer: warm gold — primary/large char
        atc='#FFC107'
        satc='#FF4081'
        # Primary position: center of key (shift layer = large)
        primary_x_offset = -0.1
        primary_y_offset = -0.1
        # Secondary position: bottom-right corner (base layer = small)
        secondary_x_offset = 0.25
        secondary_y_offset = 0.25
        alt_x_offset=-0.25
        alt_y_offset=0.25
        shift_alt_x_offset=-0.25
        shift_alt_y_offset=-0.25
        for key, data in self.keys.items():
            if all(key not in layer for layer in self.key2idx):
                continue
            rect = patches.Rectangle((data._pos.x, data._pos.y), data._width, 1, 
                         linewidth=0.1, edgecolor=ec, facecolor=fc, alpha=1)
            self.rects.append(rect)
            self.ax.add_patch(rect)

            base_char = self.remaps[0][key] if key in self.remaps[0] else None

            # Shift layer is PRIMARY — shown large at center
            shift_char = self.remaps[1][key] if key in self.remaps[1] else None
            if shift_char is None and base_char is not None and len(base_char) == 1:
                shift_char = base_char.upper()
            elif shift_char is not None:
                shift_char = shift_char.upper() if len(shift_char) == 1 else shift_char

            if shift_char is not None:
                self.texts.append(self.ax.text(data.fpos.x+primary_x_offset, data.fpos.y+primary_y_offset,
                    self._display_label(shift_char),
                    color=stc, fontsize=12, fontweight='bold',
                    ha='center', va='center'))

            # Base layer is SECONDARY — only shown if it adds info beyond the shift label
            show_base = (
                base_char is not None and
                not (len(base_char) == 1 and shift_char is not None and base_char.upper() == shift_char)
            )
            if show_base:
                self.texts.append(self.ax.text(data.fpos.x+secondary_x_offset, data.fpos.y+secondary_y_offset,
                    self._display_label(base_char),
                    color=tc, fontsize=8, fontweight='bold',
                    ha='center', va='center'))

            if len(self.key2idx) > 2 and key in self.key2idx[2]:
                alt_char = self.remaps[2][key] if key in self.remaps[2] else key
                self.texts.append(self.ax.text(data.fpos.x+alt_x_offset, data.fpos.y+alt_y_offset,
                    self._display_label(alt_char),
                    color=atc, fontsize=8, fontweight='bold',
                    ha='center', va='center'))

            if len(self.key2idx) > 3 and key in self.key2idx[3]:
                sa_char = self.remaps[3][key] if key in self.remaps[3] else key
                self.texts.append(self.ax.text(data.fpos.x+shift_alt_x_offset, data.fpos.y+shift_alt_y_offset,
                    self._display_label(sa_char),
                    color=satc, fontsize=8, fontweight='bold',
                    ha='center', va='center'))

        self.ax.autoscale_view()
        plt.savefig('output/layout.svg')

    def _init_home_keys(self):
        self.home_keys={}
        self.hand=[[],[]] # Initialize two empty lists for left and right hands
        file_name="home_keys.json"
        check_required_config_file(file_name)
        try:
            with open(f'config/{file_name}','r') as file: 
                for finger, key in json.load(file).items():
                    self._validate_finger(finger,file_name)
                    self._does_key_exist(key, self.keys, file_name, 'layout.txt')

                    found=False
                    for i,layer in enumerate(self.key2idx):
                        if key not in layer:
                            continue
                        self.home_keys[self.finger2idx[finger]]=self.key2idx[i][key]
                        found=True

                    if found:
                        hand_code, finger_code=self.get_finger_roll(self.finger2idx[finger])
                        self.hand[hand_code].append(finger_code)
                    else:
                        print(f"WARNING: Home key [{key}] for finger [{finger}] is not active (not in fixed/available keys), skipping finger.")
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
            raise SystemExit

        self.hand[0].sort()
        self.hand[1].sort()

        natural_pos={}
        file_name='finger_natural_positions.json'
        check_required_config_file(file_name)
        with open(f'config/{file_name}','r') as file:
            natural_pos=json.load(file)
        self.finger_natural_pos={}
        for finger_idx in self.home_keys:
            hand, finger=self.idx2finger[finger_idx].split('_')
            if hand not in natural_pos:
                raise ValueError(f"\nHAND [{hand.upper()}] IN {file_name} FILE IS INVALID!")
            if finger not in natural_pos[hand]['x']:
                raise ValueError(f"\nFINGER [{finger.upper()}] DOESNT APPEAR IN [{hand.upper()}]:X HAND IN {file_name} FILE!")
            if finger not in natural_pos[hand]['y']:
                raise ValueError(f"\nFINGER [{finger.upper()}] DOESNT APPEAR IN [{hand.upper()}]:Y HAND IN {file_name} FILE!")

            self.finger_natural_pos[finger_idx]=Point(natural_pos[hand]['x'][finger],natural_pos[hand]['y'][finger])

    def _init_parameters(self):
        self.finger_efforts={}
        file_name='parameters.json'
        check_required_config_file(file_name)
        try:
            with open(f'config/{file_name}','r') as file:
                parameters=json.load(file)
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
            raise SystemExit
        if "finger_efforts" not in parameters:
            raise IndexError(f"\nPLEASE ENTER FINGER EFFORTS IN {file_name} FILE FIRST!")
        
        self.skip_backspace = parameters.get("skip_backspace", False)
        self.skip_home = parameters.get("skip_home", False)
        self.xkb_filename = parameters.get("xkb_filename", "layout.xkb")
        self.roll_bonus_2 = parameters.get("roll_bonus_2", 1.5)
        self.roll_bonus_3 = parameters.get("roll_bonus_3", 2.5)

        self.finger_efforts=parameters["finger_efforts"]
        for finger in self.finger_efforts:
            if finger not in self.finger2idx:
                raise ValueError(f"\nFINGER NAME [{finger.upper()}] YOU ASSIGNED IN {file_name}:FINGER_EFFORTS FILE IS INVALID!")
        #encode
        self.finger_efforts={self.finger2idx[finger]: effort for finger, effort in self.finger_efforts.items()}

    def _init_max_finger_dists(self): #need init finger roll first
        tfinger_dists={}
        file_name='max_finger_distances.json'
        check_required_config_file(file_name)
        try:
            with open(f"config/{file_name}","r") as file:
                tfinger_dists=json.load(file)
        except json.JSONDecodeError as e:
            print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
            raise SystemExit

        self.finger_dists={} #THIS USE FINGER CODE AND HAND CODE AS INDEX SYSTEM (SAME WITH FINGER_CODE, HAND_CODE) NOT THE FINGER_IDX SYSTEM (SAME WITH finger2idx)
        for FF, dist in tfinger_dists.items():
            x, y=FF.split('_')
            assert x in self.FINGER_CODE and y in self.FINGER_CODE, f"\nFINGER NAMES [{FF.upper()}] IN {file_name} FILE ARE INVALID!"
            self.finger_dists[(self.FINGER_CODE[x],self.FINGER_CODE[y])]=dist
            self.finger_dists[(self.FINGER_CODE[y],self.FINGER_CODE[x])]=dist


    def _precompute(self):
        self.key_sq_dists=tuple([distance_sq(self.keys[self.idx2key[i]].fpos,self.keys[self.idx2key[j]].fpos) for i in range(len(self.idx2key))] for j in range(len(self.idx2key))) #SQUARE OF DISTANCE BETWEEN KEYS, USE KEY_IDX AS INDEX SYSTEM

        # PRECOMPUTING MAXIMUM DISTANCE FOR PRINCIPLED GEOMETRIC PENALTIES
        self.max_sq_dist = max(max(row) for row in self.key_sq_dists)

        #finger strain heapmap
        self.strain_heapmap=[]

        for key_idx in range(len(self.idx2key)):
            finger_idx=self.key_idx2finger_idx[key_idx]
            if finger_idx is None:
                # Key exists in layout but has no assigned finger (e.g. right-hand keys,
                # number row) — not used in optimization, assign zero strain
                self.strain_heapmap.append(0)
                continue
            anchor_dist_sq=self.key_sq_dists[key_idx][self.home_keys[finger_idx]]
            self.strain_heapmap.append(anchor_dist_sq*self.finger_efforts[finger_idx])
        self.strain_heapmap=tuple(self.strain_heapmap)
        #available only strain heapmap, cumulate them, flatten out
        FLATTEN_STRENGTH=0.5
        self.cum_avai_strain_heapmap=[]
        last_strain=0
        for k in self.available_keys:
            last_strain=self.strain_heapmap[k]**FLATTEN_STRENGTH+last_strain
            self.cum_avai_strain_heapmap.append(last_strain)

        self.cum_avai_strain_heapmap=tuple(self.cum_avai_strain_heapmap)

        #TODO: precompute FS factors



    
            
    #GETTERS
    def get_finger_roll(self, finger_idx): #THIS FUNCTION RETURNS DIFFERENT INDEX SYSTEM (SAME WITH FINGER_CODE, HAND_CODE)
        hand_code=finger_idx//5
        finger_code=finger_idx%5
        return hand_code, finger_code

    def get_finger_idx(self, hand_code, finger_code): #THIS FUNCTION TAKES IN DIFFERENT INDEX SYSTEM (SAME WITH FINGER_CODE, HAND_CODE)
        return hand_code*5+finger_code

    def display(self, potential_remaps:list, scores:tuple=(), name='layout'):
        for i in range(len(self.texts)):
            self.texts[i].remove()
        self.texts=[]
        base_x_offset=-0.1
        base_y_offset=-0.1
        tc='#EEEEEE'

        key_base = {}
        key_shift = {}
        key_alt = {}
        key_shift_alt = {}

        for keyIdx, remapIdx in enumerate(potential_remaps):
            if remapIdx is None:
                continue
            key = self.idx2key[keyIdx]
            remap = self.keybinds[remapIdx]
            if keyIdx < self.sizes[0]:
                key_base[key] = remap
            elif keyIdx < self.sizes[0] + self.sizes[1]:
                key_shift[key] = remap
            elif len(self.sizes) > 2 and keyIdx < self.sizes[0] + self.sizes[1] + self.sizes[2]:
                key_alt[key] = remap
            else:
                key_shift_alt[key] = remap

        for key, data in self.keys.items():
            if all(key not in layer for layer in self.key2idx):
                continue

            base_char = key_base.get(key)

            # Shift layer = PRIMARY: large, center position
            shift_char = key_shift.get(key)
            if shift_char is None and base_char is not None and len(base_char) == 1:
                shift_char = base_char.upper()
            elif shift_char is not None:
                shift_char = shift_char.upper() if len(shift_char) == 1 else shift_char

            if shift_char is not None:
                self.texts.append(self.ax.text(data.fpos.x + base_x_offset, data.fpos.y + base_x_offset,
                    self._display_label(shift_char),
                    color='#FFD580', fontsize=12, fontweight='bold',
                    ha='center', va='center'))

            # Base layer = SECONDARY: only shown if it differs from the implied lowercase of shift
            show_base = (
                base_char is not None and
                not (len(base_char) == 1 and shift_char is not None and base_char.upper() == shift_char)
            )
            if show_base:
                self.texts.append(self.ax.text(data.fpos.x + 0.25, data.fpos.y + 0.25,
                    self._display_label(base_char),
                    color=tc, fontsize=8, fontweight='bold',
                    ha='center', va='center'))

            # Alt layer
            alt_char = key_alt.get(key)
            if alt_char is not None:
                self.texts.append(self.ax.text(data.fpos.x - 0.25, data.fpos.y + 0.25,
                    self._display_label(alt_char),
                    color='#FFC107', fontsize=8, fontweight='bold',
                    ha='center', va='center'))

            # Shift+Alt layer
            sa_char = key_shift_alt.get(key)
            if sa_char is not None:
                self.texts.append(self.ax.text(data.fpos.x - 0.25, data.fpos.y - 0.25,
                    self._display_label(sa_char),
                    color='#FF4081', fontsize=8, fontweight='bold',
                    ha='center', va='center'))


        self.ax.autoscale_view()
        plt.savefig(f'output/{name}.svg')

    def _does_key_exist(self,key: str, container: dict, file, root_file):
        if key not in container:
            raise ValueError(f"\nKEY SLOT [{key.upper()}] YOU ASSIGNED IN {file} FILE DOESN'T APPEAR IN {root_file} FILE!")
        return True

    def _is_key_correct_type(self,key: str, file):
        if not isinstance(key,str):
            raise TypeError(f"\nKEY SLOT [{key.upper()}] YOU ASSIGNED IN {file} FILE HAS INCORRECT TYPE, IT NEEDS TO BE STRING NOT ({type(key)})!")
        return True
    def _validate_finger(self, finger, file):
        if finger not in self.finger2idx:
            raise ValueError(f"\nFINGER NAME [{finger.upper()}] YOU ASSIGNED IN {file} FILE IS INVALID!")
        return True





if __name__=="__main__":

    l=Layout()
    
    print("Layout:",list(l.keys.keys()))
    print("Shift key:", l.shift,l.shift_name)
    print("Chat key:",l.chat)
    print("Fixed keys (Multiple layers):",[(l.idx2key[key],l.keybinds[remap]) for key, remap in l.fixed_keys.items()])
    print("Remaps:", [[(key,remap) for key,remap in layer.items()] for layer in l.remaps])
    print("Available keys:", [l.idx2key[key] for key in l.available_keys])
    print("Key positions: (Multiple layers)", [(l.idx2key[i],pos) for i, pos in enumerate(l.key_positions)])
    print("Keystrokes:", [[l.keybinds[key] for key in keystroke[0]]for keystroke in l.keystrokes])
    print("Available keybinds:", [l.keybinds[key] for key in l.available_keybinds])
    print("Keybind probability:", [(l.keybinds[i],round(prob,2)) for i,prob in enumerate(l.key_probs)])
    print("Special keybinds:", [l.keybinds[key] for key in l.special_keybinds])
    print("Available special keybinds:", [l.keybinds[key] for key in l.available_skb])
    # print("Assiged finger:")
    #TODO: continue