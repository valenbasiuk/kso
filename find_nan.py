import evaluate
import classes
import init
import numpy as np

layout = classes.Layout()
from init import Parameters
paras = Parameters()
layout.target_metrics = paras.target_metrics

pop = init.init(20, layout)
evaluator = evaluate.Evaluator(layout)

# We want to find any layout or keystroke option that returns nan
for l_idx, ind in enumerate(pop):
    keybind_idx2key_idx=[None]*len(evaluator.keybinds)
    for i,j in enumerate(ind):
        if j is not None:
            keybind_idx2key_idx[j]=i
            
    shift_s=set(ind[i] for i in range(evaluator.sizes[0],evaluator.sizes[0]+evaluator.sizes[1]) if i < len(ind) and ind[i] is not None)
    alt_s=set()
    shift_alt_s=set()
    if len(evaluator.sizes) > 2:
        alt_s=set(ind[i] for i in range(evaluator.sizes[0]+evaluator.sizes[1], evaluator.sizes[0]+evaluator.sizes[1]+evaluator.sizes[2]) if i < len(ind) and ind[i] is not None)
    if len(evaluator.sizes) > 3:
        shift_alt_s=set(ind[i] for i in range(evaluator.sizes[0]+evaluator.sizes[1]+evaluator.sizes[2], len(ind)) if i < len(ind) and ind[i] is not None)

    for k_idx, keystroke_data in enumerate(evaluator.keystrokes):
        options = keystroke_data["options"]
        for opt in options:
            if any(keybind_idx2key_idx[kb] is None for kb in opt):
                continue
            nks, sd, ad = evaluator.normalize_keystroke_single(ind, opt, shift_s, alt_s, shift_alt_s)
            
            try:
                travel_dist, use_count, roll, stretch = evaluator.single_sequence_cost(ind, keybind_idx2key_idx, nks, sd, ad)
                # Check for any nan values
                if np.isnan(travel_dist) or np.isnan(use_count) or np.isnan(roll) or np.isnan(stretch):
                    print(f"NAN detected! Layout {l_idx}, Keystroke {k_idx}, Option {opt}")
                    print(f"Results: travel={travel_dist}, use={use_count}, roll={roll}, stretch={stretch}")
                    exit(0)
            except Exception as e:
                print(f"Exception for Layout {l_idx}, Keystroke {k_idx}, Option {opt}: {e}")
                exit(1)

print("No NaNs or exceptions found in fuzzer!")
