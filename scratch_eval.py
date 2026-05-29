import evaluate
import classes
import init
import numpy as np

layout = classes.Layout()
from init import Parameters
paras = Parameters()
layout.target_metrics = paras.target_metrics

pop = init.init(5, layout)
evaluator = evaluate.Evaluator(layout)

print("layout.sizes:", layout.sizes)
print("shift_only_kbs:", [layout.keybinds[x] for x in evaluator.shift_only_kbs])
print("base_only_kbs:", [layout.keybinds[x] for x in evaluator.base_only_kbs])
print("layout.available_keys count:", len(layout.available_keys))
print("Available shift keys count:", sum(1 for k in layout.available_keys if k >= layout.sizes[0]))
print("Available base keys count:", sum(1 for k in layout.available_keys if k < layout.sizes[0]))
print("shift_only_kbs count:", len(evaluator.shift_only_kbs))
ind = pop[0]
for i, ind in enumerate(pop):
    res_correct, case = evaluator.correct(ind)
    print(f"Population member {i} correctness: {res_correct}, case: {case}")
    if case == 4:
        for idx, kb in enumerate(ind):
            if kb is not None and kb in evaluator.shift_only_kbs and idx < layout.sizes[0]:
                print(f"  FAILED: keybind '{layout.keybinds[kb]}' (kb {kb}) is on base layer at '{layout.idx2key[idx]}' (idx {idx})!")
        
import init
print("Running distributed_init step by step:")
special_set = set(layout.special_keybinds)
available_specials = [kb for kb in layout.available_keybinds if kb in special_set]
print(f"available_specials: {available_specials} ({[layout.keybinds[x] for x in available_specials]})")
base_scored = []
for key_idx in layout.layered_available_keys[0]:
    finger_idx = layout.key_idx2finger_idx[key_idx]
    if finger_idx is None or finger_idx not in layout.finger_efforts or finger_idx not in layout.home_keys:
        base_scored.append((float('inf'), key_idx))
        continue
    score = layout.finger_efforts[finger_idx] * layout.key_sq_dists[key_idx][layout.home_keys[finger_idx]]
    base_scored.append((score, key_idx))
base_scored.sort(key=lambda x: x[0])
base_indices = [idx for _, idx in base_scored]
print(f"base_indices: {base_indices}")

res = [None] * len(layout.idx2key)
for idx, kb in layout.fixed_keys.items():
    res[idx] = kb

blocked = set()
specials_to_place = list(available_specials)
for pos in base_indices:
    if not specials_to_place:
        break
    if pos in blocked or res[pos] is not None:
        continue
    
    # Check if counterparts are free/fixed
    cps = layout.counterparts.get(pos, [])
    if any(cp in layout.fixed_keys or res[cp] is not None for cp in cps):
        print(f"Skipping pos {pos} because counterpart is fixed/assigned: {cps}")
        continue
        
    chosen = specials_to_place.pop(0)
    res[pos] = chosen
    print(f"Placed special {layout.keybinds[chosen]} at pos {pos} ({layout.idx2key[pos]})")
    for cp in cps:
        blocked.add(cp)

print(f"Remaining specials to place: {specials_to_place}")


