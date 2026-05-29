import numpy as np
import random
from classes import *
import random
import json
class Parameters:
    def __init__(self):
            DEFAULT_GENERATION_LIMIT = 3000
            DEFAULT_POPULATION_SIZE = 10
            DEFAULT_DEV_MODE=False
            self.target_metrics={}
            file_name="target_metrics.json"
            try:
                with open(f'config/{file_name}') as file:
                    target_metrics=json.load(file)
            except json.JSONDecodeError as e:
                print(f"\nSYNTAX ERROR IN {file_name} FILE: {e}")
                raise SystemExit
            self.target_metrics=normalize_weights(target_metrics)
    

            with open('config/parameters.json') as file:
                paras=json.load(file)
            if "generation_limit" in paras:
                self.generation_limit=paras["generation_limit"]
            else:
                self.generation_limit=DEFAULT_GENERATION_LIMIT
            if "population_size" in paras:
                self.population_size=paras["population_size"]
            else:
                self.population_size=DEFAULT_POPULATION_SIZE
            if "dev_mode" in paras:
                self.dev_mode=paras["dev_mode"]
            else:
                self.dev_mode=DEFAULT_DEV_MODE

'''
'''
def distributed_init(layout: Layout):
    # --- Phase 0: seed result with fixed keys ---
    res = [None] * len(layout.idx2key)
    for idx, kb in layout.fixed_keys.items():
        res[idx] = kb

    # --- Phase 1: partition available keybinds ---
    special_set = set(layout.special_keybinds)
    available_specials = [kb for kb in layout.available_keybinds if kb in special_set]
    available_normals  = [kb for kb in layout.available_keybinds if kb not in special_set]

    # Small random swaps for diversity (specials)
    available_specials.sort(key=lambda x: layout.key_probs[x], reverse=True)
    for i in range(len(available_specials)):
        if random.random() < 0.1:
            j = random.randint(0, len(available_specials) - 1)
            available_specials[i], available_specials[j] = available_specials[j], available_specials[i]

    # Small random swaps for diversity (normals)
    available_normals.sort(key=lambda x: layout.key_probs[x], reverse=True)
    for i in range(len(available_normals)):
        if random.random() < 0.1:
            j = random.randint(0, len(available_normals) - 1)
            available_normals[i], available_normals[j] = available_normals[j], available_normals[i]

    blocked = set()  # counterpart slots made unavailable by placed specials

    def base_score(key_idx):
        fi = layout.key_idx2finger_idx[key_idx]
        if fi is None or fi not in layout.finger_efforts or fi not in layout.home_keys:
            return float('inf')
        return layout.finger_efforts[fi] * layout.key_sq_dists[key_idx][layout.home_keys[fi]]

    # --- Phase 2: score & sort base positions for specials ---
    base_candidates = []
    for key_idx in layout.layered_available_keys[0]:
        if res[key_idx] is not None:
            continue
        cps = layout.counterparts.get(key_idx, [])
        if any(res[cp] is not None or cp in layout.fixed_keys for cp in cps):
            continue
        base_candidates.append((base_score(key_idx), key_idx))
    base_candidates.sort(key=lambda x: x[0])

    # --- Phase 3: place specials, guaranteeing enough normal slots remain ---
    def free_normal_slot_count(extra_blocked):
        combined = blocked | extra_blocked
        return sum(1 for ki in layout.available_keys if res[ki] is None and ki not in combined)

    specials_remaining = list(available_specials)
    normals_needed = len(available_normals)

    for _, pos in base_candidates:
        if not specials_remaining:
            break
        if res[pos] is not None or pos in blocked:
            continue
        cps = layout.counterparts.get(pos, [])
        if any(cp in blocked or res[cp] is not None for cp in cps):
            continue
        new_blocked = set(cps)
        free_after = free_normal_slot_count(new_blocked) - 1
        specials_left_after = len(specials_remaining) - 1
        if free_after < normals_needed + specials_left_after:
            continue
        res[pos] = specials_remaining.pop(0)
        blocked |= new_blocked

    # Fallback: force remaining specials onto any valid base slot
    for _, pos in base_candidates:
        if not specials_remaining:
            break
        if res[pos] is not None or pos in blocked:
            continue
        cps = layout.counterparts.get(pos, [])
        if any(cp in blocked or res[cp] is not None for cp in cps):
            continue
        res[pos] = specials_remaining.pop(0)
        blocked |= set(cps)

    # --- Phase 4: identify required keybinds ---
    # For each keystroke, pick the option that requires the fewest additional
    # normals to satisfy, and mark those normals as "required". This guarantees
    # correct() returns True for every craft after placement.
    already_placed = set(kb for kb in res if kb is not None)
    normals_pool_set = set(available_normals)
    required_kbs = set()

    for ks_data in layout.keystrokes:
        options = ks_data["options"]
        # Already satisfiable by fixed/special placements?
        if any(all(kb in already_placed for kb in opt) for opt in options):
            continue
        # Find option needing fewest additional normals
        best_missing = None
        for opt in options:
            missing = [kb for kb in opt if kb not in already_placed and kb in normals_pool_set]
            if best_missing is None or len(missing) < len(best_missing):
                best_missing = missing
        if best_missing:
            required_kbs.update(best_missing)

    # Split normals: required first (preserving probability-sorted order), then optional
    required_normals = [kb for kb in available_normals if kb in required_kbs]
    optional_normals = [kb for kb in available_normals if kb not in required_kbs]

    # Identify shift-only and base-only normal keybinds for layer placement
    normal_kbs = set(layout.available_keybinds) - set(layout.special_keybinds)
    kb_in_shift = set()
    kb_in_base = set()
    for ks in layout.keystrokes:
        touch_shift = ks.get("touch_shift", False)
        for opt in ks["options"]:
            for kb in opt:
                if kb in normal_kbs:
                    if touch_shift:
                        kb_in_shift.add(kb)
                    else:
                        kb_in_base.add(kb)
    num_shift_slots = sum(1 for k in layout.available_keys if k >= layout.sizes[0]) - len(layout.special_keybinds)
    sorted_shift_kbs = sorted(kb_in_shift, key=lambda x: layout.key_probs[x], reverse=True)
    shift_only_kbs = set(sorted_shift_kbs[:num_shift_slots])
    base_only_kbs = kb_in_base - shift_only_kbs

    # --- Phase 5: score all remaining available slots for normals ---
    def normal_score(key_idx):
        key_name = layout.idx2key[key_idx]
        fi = layout.key_idx2finger_idx[key_idx]
        if fi is None or fi not in layout.finger_efforts or fi not in layout.home_keys:
            return float('inf')
        layer_penalty = 1.0 if key_idx < layout.sizes[0] else 1.25
        
        # Prioritize filling core spaces (left-hand keyboard rows)
        priority_keys = {'a', 's', 'd', 'f', 'g', 'q', 'w', 'e', 'r', 't', 'z', 'x', 'c', 'v', '1', '2', '3', '4', '5'}
        priority_bonus = 0.01 if key_name in priority_keys else 10.0
        
        return layout.finger_efforts[fi] * layout.key_sq_dists[key_idx][layout.home_keys[fi]] * layer_penalty * priority_bonus

    # Separate available slots by layer candidates
    base_candidates = []
    shift_candidates = []
    for key_idx in layout.available_keys:
        if res[key_idx] is None and key_idx not in blocked:
            score = normal_score(key_idx)
            if key_idx < layout.sizes[0]:
                base_candidates.append((score, key_idx))
            else:
                shift_candidates.append((score, key_idx))
    base_candidates.sort(key=lambda x: x[0])
    shift_candidates.sort(key=lambda x: x[0])

    normals_to_place = required_normals + optional_normals
    shift_only_to_place = [kb for kb in normals_to_place if kb in shift_only_kbs]
    base_only_to_place = [kb for kb in normals_to_place if kb in base_only_kbs]
    neutral_to_place = [kb for kb in normals_to_place if kb not in shift_only_kbs and kb not in base_only_kbs]

    # Place shift-only normals on shift slots
    for kb in shift_only_to_place:
        if not shift_candidates:
            if base_candidates:
                _, pos = base_candidates.pop(0)
                res[pos] = kb
        else:
            _, pos = shift_candidates.pop(0)
            res[pos] = kb

    # Place base-only normals on base slots
    for kb in base_only_to_place:
        if not base_candidates:
            if shift_candidates:
                _, pos = shift_candidates.pop(0)
                res[pos] = kb
        else:
            _, pos = base_candidates.pop(0)
            res[pos] = kb

    # Place neutral normals on any remaining slots
    remaining_candidates = base_candidates + shift_candidates
    remaining_candidates.sort(key=lambda x: x[0])
    for kb in neutral_to_place:
        if not remaining_candidates:
            break
        _, pos = remaining_candidates.pop(0)
        res[pos] = kb

    return res

def random_init(layout:Layout):
    valid_base_keys = []
    for key_idx in layout.layered_available_keys[0]:
        cps = layout.counterparts.get(key_idx, [])
        if not any(cp in layout.fixed_keys for cp in cps):
            valid_base_keys.append(key_idx)
            
    used_indices=random.sample(valid_base_keys, len(layout.special_keybinds))

    used_indice_s=set(used_indices)
    for key_idx in used_indices:
        for cp in layout.counterparts.get(key_idx, []):
            used_indice_s.add(cp)

    new_AK=[key_idx for key_idx in layout.available_keys if key_idx not in used_indice_s]
    new_AB=[keybind_idx for keybind_idx in layout.available_keybinds if keybind_idx not in layout.special_keybinds]
    indices=random.sample(new_AK, len(new_AK))
    keys=random.sample(new_AB, min(len(new_AK), len(new_AB)))

    res=[None]*len(layout.idx2key)
    for i,keybind_idx in enumerate(layout.special_keybinds):
        res[used_indices[i]]=keybind_idx
    for idx,key in zip(indices,keys):
        res[idx]=key
    for idx,kb in layout.fixed_keys.items():
        res[idx]=kb
    return res

def normalize_weights(target_metrics):
    total_weight=sum(target_metrics.values())
    if total_weight==0:
        return
    return {name:value/total_weight for name, value in target_metrics.items()}

def init_temperature(population_size:int, min_T:float, max_T:float):
    step=(max_T/min_T)**(1/(population_size-1))
    res=[min_T]
    for i in range(1,population_size-1):
        res.append(res[-1]*step)
    res.append(max_T)
    return res

def init_weight_vectors(target_metrics, population_size:int, concentration=10.0 ):
    target_metrics=normalize_weights(target_metrics)

    weight_vector=np.array(list(target_metrics.values()))

    alpha=weight_vector*concentration

    samples=np.random.dirichlet(alpha, size=population_size-1)



    weight_vecs=list(samples)

    sorted_vecs=[weight_vector]

    while weight_vecs:
        min_i=np.argmin([np.linalg.norm(sorted_vecs[-1]-v) for v in weight_vecs])
        sorted_vecs.append(weight_vecs[min_i])
        weight_vecs[min_i], weight_vecs[-1]=weight_vecs[-1], weight_vecs[min_i]
        weight_vecs.pop()

    return sorted_vecs



def init(population_size: int, layout: Layout):
    # ind=distributed_init(layout)
    return [distributed_init(layout) for _ in range(population_size)]



if __name__=='__main__':
    from evaluate import correct
    layout=Layout()
    n=100    
    init_layout=distributed_init(layout)
    # while correct(init_layout,layout):
        # print("hello")d
        # init_layout=distributed_init(layout)
    
    print(init_layout)
    layout.display(init_layout)
    print(init_weight_vectors({1:10,3:10,2:10},10))
    