import random
import numpy as np
import math
from classes import *
class Evaluator:
    def __init__(self, layout: Layout):
        self.layout=layout

        # direct aliases for speed
        self.idx2key=layout.idx2key
        self.key_probs=layout.key_probs
        self.home_keys=layout.home_keys
        self.key_idx2finger_idx=layout.key_idx2finger_idx
        self.key_idx2finger_candidates=layout.key_idx2finger_candidates
        self.chat_i=layout.chat_i
        self.hand=layout.hand
        self.sizes=layout.sizes
        self.shift_kbi=layout.shift_kbi
        self.no_shift_variance_skb_set=layout.no_shift_variance_skb_set
        self.keystrokes=layout.keystrokes
        self.total_keybinds=layout.total_keybinds
        self.special_keybinds=layout.special_keybinds
        self.counterparts=layout.counterparts
        self.available_keybinds=layout.available_keybinds
        self.fixed_keys=layout.fixed_keys
        self.strain_heapmap=layout.strain_heapmap
        self.shift_i=layout.shift_i
        self.keys=layout.keys
        self.finger_natural_pos=layout.finger_natural_pos
        self.finger_dists=layout.finger_dists
        self.finger_efforts=layout.finger_efforts
        self.max_sq_dist=layout.max_sq_dist
        self.key_sq_dists=layout.key_sq_dists
        self.keybinds=layout.keybinds

        # Identify special keys that "break" a roll (Home, Space, Backspace)
        self.roll_breakers = set()
        if layout.home_kbi is not None:
            self.roll_breakers.add(layout.home_kbi)
        for k in layout.specials:
            kb = layout.specials[k][2]
            if kb is not None:
                self.roll_breakers.add(kb)

        self.get_finger_idx=layout.get_finger_idx
        self.get_finger_roll=layout.get_finger_roll

        # Alt layer direct aliases
        self.alt_i = getattr(layout, 'alt_i', None)
        self.alt_kbi = getattr(layout, 'alt_kbi', None)
        self.no_alt_variance_skb_set = getattr(layout, 'no_alt_variance_skb_set', set())

        try:
            with open('config/parameters.json', 'r') as file:
                paras = json.load(file)
                self.ALT_LAYER_PENALTY = paras.get("alt_layer_penalty", 1.5)
        except Exception:
            self.ALT_LAYER_PENALTY = 1.5

        self.REDIRECT_PENALTY_COEFF=4
        self.INWARD_REWARD_COEFF=3
        self.OUTWARD_REWARD_COEFF=1
        self.SAME_FINGER_PENALTY_COEFF=3
        self.TRAVEL_DISTANCE_COEFF=0.6
        self.FINGER_STRETCH_COEFF=0.6
        self.SHIFT_HOLDING_STRAIN_COEFF=0.5
        self.roll_bonus_2 = getattr(layout, 'roll_bonus_2', 1.5)
        self.roll_bonus_3 = getattr(layout, 'roll_bonus_3', 2.5)
        self.roll_bonus_4 = getattr(layout, 'roll_bonus_4', 3.5)
        self.stretch_decaying_factor=math.exp(self.FINGER_STRETCH_COEFF)
        self._precompute()

        # Reward for rolling AND being on adjacent rows simultaneously
        # (the QWE→ASD→ZXC column-flow pattern the user wants)
        self.COLUMN_FLOW_BONUS_COEFF = getattr(layout, 'column_flow_bonus', 1.2)
        # Precompute the keyboard row (integer) for each key index
        self.key_row = [
            round(self.keys[self.idx2key[i]].fpos.y)
            for i in range(len(self.idx2key))
        ]

        # Identify shift-only and base-only normal keybinds
        normal_kbs = set(self.available_keybinds) - set(self.special_keybinds)
        kb_in_shift = set()
        kb_in_base = set()
        for ks in self.keystrokes:
            touch_shift = ks.get("touch_shift", False)
            for opt in ks["options"]:
                for kb in opt:
                    if kb in normal_kbs:
                        if touch_shift:
                            kb_in_shift.add(kb)
                        else:
                            kb_in_base.add(kb)
        num_shift_slots = sum(1 for k in self.layout.available_keys if k >= self.sizes[0]) - len(self.special_keybinds)
        sorted_shift_kbs = sorted(kb_in_shift, key=lambda x: self.key_probs[x], reverse=True)
        self.shift_only_kbs = set(sorted_shift_kbs[:num_shift_slots])
        self.base_only_kbs = kb_in_base - self.shift_only_kbs

    def _precompute(self):
        #cache finger roll
        self.finger_rolls={}
        for finger_idx in self.home_keys.keys():
            self.finger_rolls[finger_idx]=self.get_finger_roll(finger_idx)
        
        max_time=0
        for keystroke in self.keystrokes:
            for opt in keystroke["options"]:
                max_time=max(max_time,len(opt))
        max_time=max_time*2+1
        self.TD_decay_cache=[1/math.exp(time*self.TRAVEL_DISTANCE_COEFF) for time in range(max_time)]
        max_time*=2
        self.FS_decay_cache=[self.stretch_decaying_factor**time for time in range(max_time)]

        self.initial_FS_cache=[{},{}] #cache for FS calculation, index by hand code, key is (finger_i, finger_j) with finger code as index system, value is the FS cost between the two fingers
        self.initial_FS_total=0
        
        #assume 1 finger is pressing chat if chat_matters is True
        if self.layout.chat_matters:
            temp_key=self.home_keys[self.key_idx2finger_idx[self.chat_i[0]]] #store the original key idx for the finger that presses chat
            self.home_keys[self.key_idx2finger_idx[self.chat_i[0]]]=self.chat_i[0]

        for hand_code in [0,1]:
            self.initial_FS_total+=self.FS_full(self.home_keys, self.hand[hand_code], hand_code, self.initial_FS_cache)

        if self.layout.chat_matters:
            self.home_keys[self.key_idx2finger_idx[self.chat_i[0]]]=temp_key #revert back

    def normalize_keystroke_single(self, ind, option_keys, shift_s, alt_s, shift_alt_s):
        shift = False
        alt = False
        nkeystroke = []
        shift_durations = []
        alt_durations = []
        last_shift_duration = 0
        last_alt_duration = 0
        shift_intended = False
        alt_intended = False
        
        alt_kbi = getattr(self, 'alt_kbi', -999)
        if alt_kbi is None:
            alt_kbi = -999

        for i, kb_idx in enumerate(option_keys):
            # Check Shift state
            if kb_idx == self.shift_kbi:
                if shift:
                    continue
                shift = True
                shift_intended = True
            elif kb_idx in shift_s or kb_idx in shift_alt_s or shift_intended:
                if not shift:
                    nkeystroke.append(self.shift_kbi)
                    shift_durations.append(1)
                    if alt_kbi != -999:
                        alt_durations.append(last_alt_duration)
                    last_shift_duration = 1
                    shift = True
                shift_intended = False
            elif not (shift and kb_idx in self.no_shift_variance_skb_set and i < len(option_keys)-1 and (option_keys[i+1] in shift_s or option_keys[i+1] in shift_alt_s or option_keys[i+1] == self.shift_kbi)):
                if shift:
                    nkeystroke.append(self.shift_kbi)
                    shift_durations.append(0)
                    if alt_kbi != -999:
                        alt_durations.append(last_alt_duration)
                    last_shift_duration = 0
                shift = False

            # Check Alt state
            if alt_kbi != -999:
                if kb_idx == alt_kbi:
                    if alt:
                        continue
                    alt = True
                    alt_intended = True
                elif kb_idx in alt_s or kb_idx in shift_alt_s or alt_intended:
                    if not alt:
                        nkeystroke.append(alt_kbi)
                        alt_durations.append(1)
                        shift_durations.append(last_shift_duration)
                        last_alt_duration = 1
                        alt = True
                    alt_intended = False
                elif not (alt and kb_idx in self.no_alt_variance_skb_set and i < len(option_keys)-1 and (option_keys[i+1] in alt_s or option_keys[i+1] in shift_alt_s or option_keys[i+1] == alt_kbi)):
                    if alt:
                        nkeystroke.append(alt_kbi)
                        alt_durations.append(0)
                        shift_durations.append(last_shift_duration)
                        last_alt_duration = 0
                    alt = False

            # Update durations
            if shift:
                last_shift_duration += 1
            else:
                last_shift_duration = 0

            if alt_kbi != -999:
                if alt:
                    last_alt_duration += 1
                else:
                    last_alt_duration = 0

            nkeystroke.append(kb_idx)
            shift_durations.append(last_shift_duration)
            if alt_kbi != -999:
                alt_durations.append(last_alt_duration)

        return nkeystroke, shift_durations, alt_durations

    def select_best_option_for_item(self, ind, options, weight, shift_s, alt_s, shift_alt_s, keybind_idx2key_idx):
        best_opt = None
        best_cost = float('inf')
        best_nkeystroke = None
        best_shift_durations = None
        best_alt_durations = None
        best_shift_w_contrib = 0
        best_alt_w_contrib = 0

        weights = self.layout.target_metrics or {
            "travel_distance": 1.0,
            "use_count": 1.0,
            "bad_roll": 1.0,
            "finger_stretch": 1.0,
            "finger_strain": 1.0
        }
        alt_kbi = getattr(self, 'alt_kbi', -999)
        if alt_kbi is None:
            alt_kbi = -999

        for opt in options:
            if any(keybind_idx2key_idx[kb] is None for kb in opt):
                continue

            nkeystroke, shift_durations, alt_durations = self.normalize_keystroke_single(ind, opt, shift_s, alt_s, shift_alt_s)
            
            # calculate local costs for this option
            travel_dist, use_count, roll, stretch = self.single_sequence_cost(ind, keybind_idx2key_idx, nkeystroke, shift_durations, alt_durations)
            
            # calculate shift_w and alt_w contributions
            shift_contrib = sum(weight for k in nkeystroke if k == self.shift_kbi)
            alt_contrib = sum(weight for k in nkeystroke if k == alt_kbi)
            
            # estimate local strain
            local_strain = 0
            for kb in nkeystroke:
                key_idx = keybind_idx2key_idx[kb]
                if key_idx is not None:
                    local_strain += self.strain_heapmap[key_idx]
            local_strain *= weight
            
            # Compute weighted sum
            cost = 0
            cost += weights.get("travel_distance", 0) * (travel_dist * weight)
            cost += weights.get("use_count", 0) * (use_count * weight)
            cost += weights.get("bad_roll", 0) * (roll * weight)
            cost += weights.get("finger_stretch", 0) * (stretch * weight)
            cost += weights.get("finger_strain", 0) * local_strain
            
            # Alt layer penalty for keys placed on Alt or Shift+Alt
            any_alt = any(keybind_idx2key_idx[kb] is not None and keybind_idx2key_idx[kb] >= self.sizes[0] + self.sizes[1] for kb in opt)
            if any_alt:
                cost += self.ALT_LAYER_PENALTY * weight

            if cost < best_cost:
                best_cost = cost
                best_opt = opt
                best_nkeystroke = nkeystroke
                best_shift_durations = shift_durations
                best_alt_durations = alt_durations
                best_shift_w_contrib = shift_contrib
                best_alt_w_contrib = alt_contrib

        return best_nkeystroke, best_shift_durations, best_alt_durations, best_shift_w_contrib, best_alt_w_contrib

    def normalize_keystrokes(self, ind: list):
        shift_s=set(ind[i] for i in range(self.sizes[0],self.sizes[0]+self.sizes[1]) if i < len(ind) and ind[i] is not None)
        alt_s=set()
        shift_alt_s=set()
        if len(self.sizes) > 2:
            alt_s=set(ind[i] for i in range(self.sizes[0]+self.sizes[1], self.sizes[0]+self.sizes[1]+self.sizes[2]) if i < len(ind) and ind[i] is not None)
        if len(self.sizes) > 3:
            shift_alt_s=set(ind[i] for i in range(self.sizes[0]+self.sizes[1]+self.sizes[2], len(ind)) if i < len(ind) and ind[i] is not None)
            
        nkeystrokes=[]
        shift_w=0
        alt_w=0

        keybind_idx2key_idx=[None]*len(self.keybinds)
        for i,j in enumerate(ind):
            if j is not None:
                keybind_idx2key_idx[j]=i

        for i, keystroke_data in enumerate(self.keystrokes):
            options = keystroke_data["options"]
            weight = keystroke_data["weight"]
            
            nkeystroke, shift_durations, alt_durations, shift_contrib, alt_contrib = self.select_best_option_for_item(
                ind, options, weight, shift_s, alt_s, shift_alt_s, keybind_idx2key_idx
            )
            if nkeystroke is None:
                # If chat is disabled, chat_i is not mapped and we could get None here if it contains chat.
                # Since we already filtered chat in classes.py, this shouldn't happen, but let's be robust
                # by fallback-evaluating if any options exist at all.
                continue
            
            nkeystrokes.append((nkeystroke, shift_durations, alt_durations))
            shift_w += shift_contrib
            alt_w += alt_contrib

        if not nkeystrokes:
            return None, None

        #recal the probability
        total = self.total_keybinds + shift_w + alt_w
        if total == 0:
            total = 1.0
        nkey_probs = [0.0] * len(self.key_probs)
        alt_kbi = getattr(self, 'alt_kbi', -999)
        if alt_kbi is None:
            alt_kbi = -999
            
        for i in range(len(self.key_probs)):
            if i != self.shift_kbi and i != alt_kbi:
                nkey_probs[i] = self.key_probs[i] * self.total_keybinds / total
        nkey_probs[self.shift_kbi] = shift_w / total
        if alt_kbi != -999:
            nkey_probs[alt_kbi] = alt_w / total

        return nkeystrokes, nkey_probs

    def correct(self, ind):
        s=set()
        for j,i in enumerate(ind):
            if i is not None:
                if i in s:
                    return False, 2
                s.add(i)
                if i in self.special_keybinds:
                    if j>=self.sizes[0]:
                        return False, 0
                    cps = self.counterparts.get(j, [])
                    if any(cp < len(ind) and ind[cp] is not None for cp in cps):
                        return False, 0
                
                # Check layer constraints for normal keybinds
                if i in self.shift_only_kbs:
                    if j < self.sizes[0]:
                        return False, 4  # Shift-only keybind placed on base layer
                elif i in self.base_only_kbs:
                    if j >= self.sizes[0]:
                        return False, 5  # Base-only keybind placed on shift/alt layer

        expected_unique = min(len(self.layout.available_keys), len(self.layout.available_keybinds)) + len(set(self.layout.fixed_keys.values()))
        if len(s)!=expected_unique:
            return False, 1

        for keystroke_data in self.keystrokes:
            options = keystroke_data["options"]
            has_valid = False
            for opt in options:
                # If chat is disabled, chat_name may not be in s, but it was already filtered in classes.py.
                # Just to be sure, check if we map the valid options
                if all(kb in s for kb in opt):
                    has_valid = True
                    break
            if not has_valid:
                return False, 3

        return True, -1

    def finger_strain(self, ind, nkey_probs):
        res = 0
        total_shift_hold_prob=0
        total_alt_hold_prob=0
        for i, j in enumerate(ind):
            if j is None:
                continue
            if self.sizes[0] <= i < self.sizes[0] + self.sizes[1] and i != self.shift_i[0]:
                total_shift_hold_prob+=nkey_probs[j]
            elif len(self.sizes) > 2 and self.sizes[0] + self.sizes[1] <= i < self.sizes[0] + self.sizes[1] + self.sizes[2] and (self.alt_i is None or not self.alt_i or i != self.alt_i[0]):
                total_alt_hold_prob+=nkey_probs[j]
            elif len(self.sizes) > 3 and i >= self.sizes[0] + self.sizes[1] + self.sizes[2]:
                total_shift_hold_prob+=nkey_probs[j]
                total_alt_hold_prob+=nkey_probs[j]

            res += self.strain_heapmap[i]* nkey_probs[j] 

        #add bonus for shift
        res+=self.strain_heapmap[self.shift_i[0]]*total_shift_hold_prob*self.SHIFT_HOLDING_STRAIN_COEFF
        #add bonus for alt
        if self.alt_i:
            res+=self.strain_heapmap[self.alt_i[0]]*total_alt_hold_prob*self.SHIFT_HOLDING_STRAIN_COEFF

        return res

    #TODO: fine-tuning
    def compute_TD(self, finger_cost, sq_dist, time, prev_time):
        delta_time = time - prev_time+1
        return (
            finger_cost
            *(sq_dist**2.25)
            *self.TD_decay_cache[delta_time] #fine-tune
        )

    #finger distance
    def compute_FS(self,key_i_pos, key_j_pos, natural_i, natural_j, finger_dist, finger_cost_i, finger_cost_j):
        return (
            (   
                ((key_i_pos.y-key_j_pos.y)-(natural_i.y-natural_j.y))**2
                +((key_i_pos.x-key_j_pos.x)-(natural_i.x-natural_j.x))**2
            )
            /finger_dist**2
            *finger_cost_i
            *finger_cost_j
        )

    def FS_full(self, finger_tasks, order, hand_code, cache): #finger_tasks only have 1 value for key_idx, not the time and count, since it's only used for initialization
        res=0
        for finger_i in order: #THIS USE FINGER CODE AND HAND CODE AS INDEX SYSTEM (SAME WITH FINGER_CODE, HAND_CODE) NOT THE FINGER_IDX SYSTEM (SAME WITH finger2idx)
            finger_idx_i=self.get_finger_idx(hand_code, finger_i) #REVERT BACK TO FINGER_IDX SYSTEM FOR FINGER COST
            key_i=self.idx2key[finger_tasks[finger_idx_i]]
            natural_i=self.finger_natural_pos[finger_idx_i]
            for finger_j in order:
                if finger_j<=finger_i:
                    continue
                finger_idx_j=self.get_finger_idx(hand_code, finger_j) #REVERT
                key_j=self.idx2key[finger_tasks[finger_idx_j]]
                natural_j=self.finger_natural_pos[finger_idx_j]
                cost=self.compute_FS(
                    self.keys[key_i].fpos,
                    self.keys[key_j].fpos,
                    natural_i,
                    natural_j,
                    self.finger_dists[(finger_i,finger_j)], #FINGER DISTANCE USE FINGER CODE AS INDEX SYSTEM
                    self.finger_efforts[finger_idx_i],
                    self.finger_efforts[finger_idx_j]
                )
                cache[hand_code][(finger_i,finger_j)]=cost
                res+=cost
        return res

    def FS_partial(self, finger_tasks, key_idx, finger_i, finger_idx_i, time, order, hand_code, old_res, cache): #finger task still stores old key
        key_i=self.idx2key[key_idx] #key str
        natural_i=self.finger_natural_pos[finger_idx_i]
        res=old_res
        for finger_j in order:
            if finger_i==finger_j:
                continue
            finger_idx_j=self.get_finger_idx(hand_code, finger_j) #REVERT
            key_j=self.idx2key[finger_tasks[finger_idx_j][0]] #key str
            natural_j=self.finger_natural_pos[finger_idx_j]

            order_pair=(min(finger_i,finger_j),max(finger_i,finger_j))

            res-=cache[hand_code][order_pair]*self.FS_decay_cache[finger_tasks[finger_idx_i][1]+finger_tasks[finger_idx_j][1]]
            cost=self.compute_FS(
                self.keys[key_i].fpos,
                self.keys[key_j].fpos,
                natural_i,
                natural_j,
                self.finger_dists[(finger_i,finger_j)],
                self.finger_efforts[finger_idx_i],
                self.finger_efforts[finger_idx_j],
            )
            cache[hand_code][order_pair]=cost

            res+=cost* self.FS_decay_cache[time+ finger_tasks[finger_idx_j][1]]
        return res

    def single_sequence_cost(self, ind, keybind_idx2key_idx, keystroke, shift_durations, alt_durations):
        #finger: [key_idx, last_used, use_count]
        finger_tasks={finger_idx:[key_idx,0,0] for finger_idx, key_idx in self.home_keys.items()}

        if self.layout.chat_matters:
            #assume 1 finger is pressing chat
            finger_tasks[self.key_idx2finger_idx[self.chat_i[0]]][0]=self.chat_i[0]

        #init 3 local costs
        local_travel_distance=0
        local_roll_cost=0
        local_finger_stretch=0  # accumulate over time

        #inititize for roll
        roll_state=-1
        consecutive_roll=0
        roll_row=-1
        roll_is_straight=True
        prev_key_idx_in_seq=None  # tracks last key pressed (any hand) for column flow bonus
        if self.layout.chat_matters:
            prev_hand_code, prev_finger_code=self.finger_rolls[self.key_idx2finger_idx[self.chat_i[0]]]
            prev_key_idx_in_seq=self.chat_i[0]
        else:
            prev_hand_code, prev_finger_code=None, None

        FS_cache=[dict(self.initial_FS_cache[0]),dict(self.initial_FS_cache[1])]
        current_FS_total=self.initial_FS_total

        #keep track of modifier fingers
        pressing_shift=None
        pressing_alt=None

        alt_i = getattr(self, 'alt_i', None)

        for j, keybind_idx in enumerate(keystroke):
            time=j+1
            key_idx=keybind_idx2key_idx[keybind_idx]
            if key_idx is None:
                continue

            # Resolve which finger presses this key.
            # For unassigned keys (e.g. number row) skip them entirely.
            # For shared keys pick the candidate with smallest travel distance.
            candidates = self.key_idx2finger_candidates[key_idx]
            if not candidates:
                # Key has no assigned finger — skip (contributes no cost).
                continue
            if len(candidates) == 1:
                press_finger = candidates[0]
            else:
                # Multi-candidate: pick the finger that is currently closest
                # (smallest travel distance from its last position to key_idx).
                best_finger = candidates[0]
                best_sq = self.key_sq_dists[key_idx][finger_tasks[candidates[0]][0]]
                for cand in candidates[1:]:
                    if cand not in finger_tasks:
                        continue
                    sq = self.key_sq_dists[key_idx][finger_tasks[cand][0]]
                    if sq < best_sq:
                        best_sq = sq
                        best_finger = cand
                press_finger = best_finger

            if key_idx==self.shift_i[0]: #shift action
                if pressing_shift is None: #Start pressing shift
                    pressing_shift=press_finger
                else: #Stop pressing shift
                    pressing_shift=None

            if alt_i is not None and len(alt_i) > 0 and key_idx==alt_i[0]: #alt action
                if pressing_alt is None: #Start pressing alt
                    pressing_alt=press_finger
                else: #Stop pressing alt
                    pressing_alt=None

            prev_key_idx=finger_tasks[press_finger][0]
            prev_time=finger_tasks[press_finger][1]

            is_modifier_finger = (
                (press_finger==pressing_shift and shift_durations[j]>1) or
                (pressing_alt is not None and press_finger==pressing_alt and alt_durations is not None and len(alt_durations) > j and alt_durations[j]>1)
            )

            local_travel_distance+=self.compute_TD(
                self.finger_efforts[press_finger],
                self.max_sq_dist if is_modifier_finger else
                self.key_sq_dists[key_idx][prev_key_idx],
                time,
                prev_time
            )

            # Massive penalty for row jumps on the same finger
            if not is_modifier_finger:
                row_diff = abs(self.key_row[key_idx] - self.key_row[prev_key_idx])
                if row_diff >= 3:
                    local_travel_distance += 1500 * (row_diff - 1) * self.finger_efforts[press_finger]
                elif row_diff == 2:
                    local_travel_distance += 400 * self.finger_efforts[press_finger]

            #ROLL HANDLING
            hand_code, finger_code=self.finger_rolls[press_finger]

            if prev_hand_code is None or prev_hand_code!=hand_code:
                roll_state=-1
                consecutive_roll=0
                roll_row=self.key_row[key_idx]
                roll_is_straight=True
            else:
                if finger_code > prev_finger_code: current_dir = 2
                elif finger_code < prev_finger_code: current_dir = 1
                else: current_dir = 0

                if current_dir!=roll_state and roll_state!=-1:
                    local_roll_cost += self.REDIRECT_PENALTY_COEFF*self.finger_efforts[press_finger]
                    consecutive_roll = 0
                    roll_row=self.key_row[key_idx]
                    roll_is_straight=True
                else:
                    consecutive_roll+=1
                    if consecutive_roll == 1:
                        # 2-key roll starting
                        roll_row = self.key_row[prev_key_idx_in_seq]
                        if self.key_row[key_idx] != roll_row:
                            roll_is_straight = False
                        else:
                            roll_is_straight = True
                    else:
                        # 3-key or 4-key roll
                        if self.key_row[key_idx] != roll_row:
                            roll_is_straight = False

                    # If consecutive_roll >= 2 and it's not a straight roll, reset it
                    if consecutive_roll >= 2 and not roll_is_straight:
                        consecutive_roll = 1
                        roll_row = self.key_row[prev_key_idx_in_seq]
                        if self.key_row[key_idx] != roll_row:
                            roll_is_straight = False
                        else:
                            roll_is_straight = True

                    if roll_is_straight:
                        if current_dir==2:  # inward roll
                            local_roll_cost -= self.INWARD_REWARD_COEFF*consecutive_roll/self.finger_efforts[press_finger]
                            # Bonus for completing a clean 2/3/4-key roll
                            if consecutive_roll == 1:
                                local_roll_cost -= self.roll_bonus_2 / self.finger_efforts[press_finger]
                            elif consecutive_roll == 2:
                                local_roll_cost -= self.roll_bonus_3 / self.finger_efforts[press_finger]
                            elif consecutive_roll >= 3:
                                local_roll_cost -= self.roll_bonus_4 / self.finger_efforts[press_finger]
                        elif current_dir==1:  # outward roll
                            local_roll_cost -= self.OUTWARD_REWARD_COEFF*consecutive_roll/self.finger_efforts[press_finger]
                            if consecutive_roll == 1:
                                local_roll_cost -= self.roll_bonus_2 / self.finger_efforts[press_finger]
                            elif consecutive_roll == 2:
                                local_roll_cost -= self.roll_bonus_3 / self.finger_efforts[press_finger]
                            elif consecutive_roll >= 3:
                                local_roll_cost -= self.roll_bonus_4 / self.finger_efforts[press_finger]
                    elif current_dir==0:
                        local_roll_cost += self.SAME_FINGER_PENALTY_COEFF*self.finger_efforts[press_finger]*consecutive_roll*2

                    # Column-flow bonus: horizontal roll on same row (straight only)
                    if current_dir in [1, 2] and prev_key_idx_in_seq is not None:
                        row_diff = abs(self.key_row[key_idx] - self.key_row[prev_key_idx_in_seq])
                        if row_diff == 0:
                            local_roll_cost -= self.COLUMN_FLOW_BONUS_COEFF / self.finger_efforts[press_finger]

                roll_state=current_dir
            prev_hand_code=hand_code
            prev_finger_code=finger_code
            prev_key_idx_in_seq=key_idx

            # Break the roll context if we just pressed a special key (e.g. Space, Backspace, Home)
            if keybind_idx in self.roll_breakers:
                prev_hand_code = None
                prev_finger_code = None

            # 1. FS for pressing finger
            current_FS_total=(
                self.FS_partial(finger_tasks, key_idx, finger_code, press_finger, time, self.hand[hand_code], hand_code, current_FS_total, FS_cache)
            )

            # 2. Update pressing finger
            finger_tasks[press_finger][0]=key_idx            
            finger_tasks[press_finger][1]=time
            finger_tasks[press_finger][2]+=1

            # 3. If shift held by different finger, FS for shift too
            if pressing_shift is not None and pressing_shift != press_finger:
                shift_hand_code, shift_finger_code = self.finger_rolls[pressing_shift]
                shift_key_idx = finger_tasks[pressing_shift][0]
                current_FS_total = self.FS_partial(finger_tasks, shift_key_idx, shift_finger_code, pressing_shift, time, self.hand[shift_hand_code], shift_hand_code, current_FS_total, FS_cache)

                finger_tasks[pressing_shift][1]=time
                finger_tasks[pressing_shift][2]+=0.5

            # 4. If alt held by different finger, FS for alt too
            if pressing_alt is not None and pressing_alt != press_finger:
                alt_hand_code, alt_finger_code = self.finger_rolls[pressing_alt]
                alt_key_idx = finger_tasks[pressing_alt][0]
                current_FS_total = self.FS_partial(finger_tasks, alt_key_idx, alt_finger_code, pressing_alt, time, self.hand[alt_hand_code], alt_hand_code, current_FS_total, FS_cache)

                finger_tasks[pressing_alt][1]=time
                finger_tasks[pressing_alt][2]+=0.5

            # global decay
            local_finger_stretch += current_FS_total/(self.FS_decay_cache[2*time])

        #cost of moving all finger back to home row
        time=len(keystroke)
        if pressing_shift is not None:
            finger_tasks[pressing_shift][1]=time-1
        if pressing_alt is not None:
            finger_tasks[pressing_alt][1]=time-1
        for finger, key_idx_n_time_n_count in finger_tasks.items():
            prev_key_idx=key_idx_n_time_n_count[0]
            prev_time=key_idx_n_time_n_count[1]

            key_idx=self.home_keys[finger]
            if prev_key_idx==key_idx:
                continue
            local_travel_distance+=self.compute_TD(
                self.finger_efforts[finger],
                self.key_sq_dists[key_idx][prev_key_idx],
                time,
                prev_time
            )

        #use count
        local_use_count_cost=0
        for finger, key_idx_n_time_n_count in finger_tasks.items():
            local_use_count_cost+=(
                key_idx_n_time_n_count[2]**2
                *self.finger_efforts[finger]
            )

        return local_travel_distance, local_use_count_cost, local_roll_cost, local_finger_stretch

    def sequence_costs(self, ind: list, nkeystrokes):
        travel_distance_cost=0
        roll_cost=0
        use_count_cost=0
        finger_stretch_cost=0

        keybind_idx2key_idx=[None]*len(self.keybinds)
        for i,j in enumerate(ind):
            if j is not None:
                keybind_idx2key_idx[j]=i

        for i in range(len(nkeystrokes)):
            keystroke, shift_durations, alt_durations = nkeystrokes[i]
            weight=self.keystrokes[i]["weight"]
            
            travel_dist, use_count, roll, stretch = self.single_sequence_cost(ind, keybind_idx2key_idx, keystroke, shift_durations, alt_durations)
            
            use_count_cost+=use_count*weight
            travel_distance_cost+=travel_dist*weight
            roll_cost+=roll*weight
            finger_stretch_cost+=stretch*weight
            
        return travel_distance_cost, use_count_cost, roll_cost, finger_stretch_cost



    def evaluate(self, population):
        evas=[]
        for i in range(len(population)):
            correct, case = self.correct(population[i])
            if not correct:
                eva = {
                    "finger_strain": 100000.0,
                    "travel_distance": 100000.0,
                    "use_count": 100000.0,
                    "bad_roll": 100000.0,
                    "finger_stretch": 100000.0
                }
                evas.append(eva)
                continue

            nkeystrokes, nkey_probs=self.normalize_keystrokes(population[i])
            if nkeystrokes is None or not nkeystrokes:
                # Fill with dummy empty list so it evaluates rather than giving 999999
                nkeystrokes = []
                nkey_probs = [1.0 / len(self.key_probs)] * len(self.key_probs)
            
            travel_distance_cost, use_count_cost, roll_cost, finger_stretch_cost=self.sequence_costs(population[i], nkeystrokes)
            eva={
                "finger_strain":self.finger_strain(population[i], nkey_probs),
                "travel_distance":travel_distance_cost,
                "use_count":use_count_cost,
                "bad_roll": roll_cost,
                "finger_stretch": finger_stretch_cost
                }
            evas.append(eva)
        return evas

# target=[Point(0,0.5),Point(0.5,0.5),Point(0.6,0.5),Point(1,0.5)]
# print(evaluate([[1,0,0,0,0,0,0,0,0,0],],target))


if __name__ == '__main__':
    # from init import *
    # l=Layout()
    # i=distributed_init(l)
    # l.display(i)
    # e=Evaluator(l)
    # print(e.evaluate([i]))
    import json
    layout=Layout()
    from init import Parameters
    paras = Parameters()
    layout.target_metrics = paras.target_metrics
    file_name='base_line.json'
    check_required_config_file(file_name)
    config={}
    try:
        with open(f"config/{file_name}","r", encoding='utf-8') as file:
            config=json.load(file)
    except json.JSONDecodeError as e:
        print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
        raise SystemExit

    if "base" not in config:
        raise ValueError(f"\nYOUR {file_name} FILE NEEDS A BASE LAYER!")
    
    ind=[None]*len(layout.idx2key)

    for key, kb in config["base"].items():
        layout._does_key_exist(key,layout.keys,file_name+":BASE",'layout.txt')
        layout._does_key_exist(kb, layout.keybind2idx,file_name+":BASE",'keystrokes.json')

        ind[layout.key2idx[0][key]]=layout.keybind2idx[kb]
    
    if "shift" in config:
        for key, kb in config["shift"].items():
            layout._does_key_exist(key,layout.keys,file_name+":SHIFT",'layout.txt')
            layout._does_key_exist(kb, layout.keybind2idx,file_name+":SHIFT",'keystrokes.json')

            ind[layout.key2idx[1][key]]=layout.keybind2idx[kb]
    
    for ki, kbi in layout.fixed_keys.items():
        if ind[ki] is not None and kbi!=ind[ki]:
            raise ValueError(f"\nYOU CAN'T ASSIGN DIFFERENT VALUE FOR FIXED KEY {layout.idx2key[ki]} IN {file_name}:SHIFT!")
        ind[ki]=kbi
    evaluator=Evaluator(layout)


    correct, case=evaluator.correct(ind)
    if not correct:
        if case==0:
            raise ValueError(f"\nYOUR LAYOUT IN {file_name} FILE VIOLATES ANY OF THESE CONSTRAINTS: SHIFT SAFE, SPECIAL IN SHIFT!")
        if case==1:
            raise ValueError(f"\nYOUR LAYOUT IN {file_name} FILE DOESN'T HAVE ENOUGH KEYS/HAS REDUNDANT KEYS DECLARED IN keystrokes.json FILE!")
        if case==2:
            raise ValueError(f"\nYOUR LAYOUT IN {file_name} FILE HAS DUPLICATE KEYS")
    print("EXPECTED CRAFTS:")
    keystrokes=evaluator.normalize_keystrokes(ind)[0]
    for keystroke,_ in keystrokes:
        print([layout.keybinds[kbi] for kbi in keystroke])

    score=evaluator.evaluate([ind])[0]
    print("SCORE:",score)
    layout.display(ind,score.items(),'baseline')
    
    



def apply_scale(evaluation, mins, maxs):
    denom = np.array(maxs, dtype=float) - np.array(mins, dtype=float)
    denom[denom == 0] = 1.0  # avoid NaN when all evals are identical on a metric
    return (evaluation - np.array(mins, dtype=float)) / denom

def weighted_sum_scaled(scaled_eval: np.ndarray, target_vector: np.ndarray):
    return np.sum(scaled_eval*target_vector)

def update_global(global_extreme, evas, mode='min'):
    range_changed=False
    for e in evas:
        if any(v >= 100000.0 for v in e):
            continue
        for i,v in enumerate(e):
            if mode == 'min' and v < global_extreme[i]:
                global_extreme[i] = v
                range_changed=True
            elif mode == 'max' and v > global_extreme[i]:
                global_extreme[i] = v
                range_changed=True
    return range_changed