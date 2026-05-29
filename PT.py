import heapq
import init
import selection
import classes
import evaluate
import random
import numpy as np
import mutation
import crossover
import os
import json
from concurrent.futures import ProcessPoolExecutor


# ─── Worker globals ──────────────────────────────────────────────────────────
_worker_evaluator = None

def _init_worker(layout):
    """Each process builds its own Evaluator so __init__ caches run once."""
    global _worker_evaluator
    _worker_evaluator = evaluate.Evaluator(layout)

def _eval_chunk(chunk):
    """Evaluate a batch of replicas/candidates. Returns list of np vectors."""
    evas = _worker_evaluator.evaluate(chunk)
    return [np.array(list(e.values())) for e in evas]


# ─── Main PT loop ────────────────────────────────────────────────────────────
def run():
    paras = init.Parameters()
    objectives = list(paras.target_metrics.keys())

    min_T = 0.005
    max_T = 0.09
    n_swaps = 11  # Always odd

    layout = classes.Layout()
    layout.target_metrics = paras.target_metrics
    mutator = mutation.Mutator(layout)
    decoder = crossover.RKPosDecoder(layout)

    if paras.population_size < 1:
        return

    temperatures = init.init_temperature(paras.population_size, min_T, max_T)
    replicas = init.init(paras.population_size, layout)
    weight_vectors = init.init_weight_vectors(paras.target_metrics, paras.population_size, 50)

    # ─── Multiprocessing setup ───────────────────────────────────────────────
    n_workers = min(os.cpu_count() or 1, paras.population_size)
    # Chunk size: divide population evenly across workers
    chunk_size = max(1, paras.population_size // n_workers)

    def split(pop):
        return [pop[i:i + chunk_size] for i in range(0, len(pop), chunk_size)]

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(layout,)
    ) as pool:

        # ─── Initial evaluation ──────────────────────────────────────────────
        evaluations = [
            ev for chunk in pool.map(_eval_chunk, split(replicas))
            for ev in chunk
        ]

        valid_evas = [e for e in evaluations if all(v < 100000.0 for v in e)]
        if not valid_evas:
            valid_evas = evaluations
        dynamic_min = [min(e[i] for e in valid_evas) for i in range(len(objectives))]
        dynamic_max = [max(e[i] for e in valid_evas) for i in range(len(objectives))]

        scaled_evals = [evaluate.apply_scale(e, dynamic_min, dynamic_max) for e in evaluations]
        scores = [evaluate.weighted_sum_scaled(scaled_evals[i], weight_vectors[i])
                  for i in range(paras.population_size)]

        elites = [(scores[0], evaluations[0], replicas[0])]
        num_elites = 5

        total_jump_accept = np.zeros(paras.population_size)
        total_swap_accept = np.zeros(paras.population_size - 1)
        total_better_jumps = np.zeros(paras.population_size)

        range_changed = False
        generation_count = 0
        log_range = 1

        while generation_count < paras.generation_limit:
            generation_count += 1
            if generation_count % log_range == 0 or paras.dev_mode:
                if generation_count >= log_range * 10 and log_range < 100:
                    log_range *= 10
                print(f"\tGENERATION {generation_count}")

            # ─── Generate candidates ───────────────────────────────────────────
            candidates = []
            for i in range(paras.population_size):
                if i < paras.population_size - 1 and random.random() < 0.05:
                    p1 = replicas[i]
                    p2 = replicas[i + 1]
                    candidates.append(
                        decoder.decode(crossover.uniform_crossover_3d(decoder.encode(p1),
                                                                        decoder.encode(p2)))
                    )
                else:
                    candidates.append(mutator.mutate(replicas[i], temperatures[i]))

            # ─── Parallel candidate evaluation ─────────────────────────────────
            candidate_evaluations = [
                ev for chunk in pool.map(_eval_chunk, split(candidates))
                for ev in chunk
            ]

            # ─── Update dynamic ranges ────────────────────────────────────────
            range_changed = evaluate.update_global(dynamic_min, candidate_evaluations, 'min')
            range_changed = evaluate.update_global(dynamic_max, candidate_evaluations, 'max') or range_changed

            scaled_candidate_evals = [
                evaluate.apply_scale(e, dynamic_min, dynamic_max) for e in candidate_evaluations
            ]
            candidate_scores = [
                evaluate.weighted_sum_scaled(scaled_candidate_evals[i], weight_vectors[i])
                for i in range(paras.population_size)
            ]

            if range_changed:
                scaled_evals = [
                    evaluate.apply_scale(e, dynamic_min, dynamic_max) for e in evaluations
                ]
                scores = [
                    evaluate.weighted_sum_scaled(scaled_evals[i], weight_vectors[i])
                    for i in range(paras.population_size)
                ]
                scaled_elite_evals = [
                    evaluate.apply_scale(e, dynamic_min, dynamic_max) for _, e, _ in elites
                ]
                elites = [
                    (evaluate.weighted_sum_scaled(scaled_elite_evals[i], weight_vectors[0]),
                     elites[i][1], elites[i][2])
                    for i in range(len(elites))
                ]

            range_changed = False

            # ─── Selection (sequential, cheap) ─────────────────────────────────
            replicas, evaluations, scores, elites, jump_accept, better_jumps = \
                selection.selection(temperatures, replicas, evaluations, scores,
                                    candidates, candidate_evaluations, candidate_scores,
                                    elites, num_elites, paras.population_size,
                                    generation_count)

            # ─── Replica swaps (──────────────────────────────────────────────────
            if generation_count % n_swaps == 0:
                replicas, evaluations, scores, swap_accept = \
                    mutation.swap_replicas(temperatures, replicas, evaluations,
                                           scores, generation_count)
                total_swap_accept += swap_accept
                if paras.dev_mode:
                    print(f"SWAP ACCEPTED: {swap_accept}")

            total_jump_accept += jump_accept
            total_better_jumps += better_jumps

            if paras.dev_mode:
                print(f"JUMP ACCEPTED: {jump_accept}")
                print(f"IDEA POINT: {[round(float(v), 2) for v in dynamic_min]}")
            if generation_count % log_range == 0 or paras.dev_mode:
                print(f"BEST SCORE: {round(min(elites, key=lambda e: e[0])[0], 3)}")

    # ─── Final output ─────────────────────────────────────────────────────────
    print('-\t' * 10)
    print("\tfinished!")

    if paras.dev_mode:
        for i, replica in enumerate(replicas):
            layout.display(replica, zip(objectives, evaluations[i]), name=f'replica_{i}')

    objective_str = "\t".join(objectives)
    print(f"\tscore\t{objective_str}")
    sorted_elites = sorted(elites, key=lambda e: e[0])
    for i, elite in enumerate(sorted_elites):
        score_str = "\t\t".join(f"{round(v, 3)}" for v in elite[1])
        print(f"rank {i + 1}:\t{round(elite[0], 3)}\t{score_str}")
        if paras.dev_mode:
            binds = {"base": {}, "shift": {}, "alt": {}, "shift_alt": {}}
            for idx, kbi in enumerate(elite[2]):
                if kbi is None:
                    continue
                if idx < layout.sizes[0]: layer = "base"
                elif idx < layout.sizes[0] + layout.sizes[1]: layer = "shift"
                elif len(layout.sizes) > 2 and idx < layout.sizes[0] + layout.sizes[1] + layout.sizes[2]: layer = "alt"
                else: layer = "shift_alt"
                binds[layer][layout.idx2key[idx]] = layout.keybinds[kbi]
            print(json.dumps(binds, ensure_ascii=False))
        print()
        layout.display(elite[2], zip(objectives, elite[1]), name=f'top_{i + 1}')

    # Generate the XKB file for each elite layout (saved to output/top_N.xkb)
    for i, elite in enumerate(sorted_elites):
        out_xkb = os.path.join("output", f"top_{i + 1}.xkb")
        save_xkb_file(layout, elite, out_xkb)

    # Save all elite layouts to a file so the GUI can parse and display them
    best_layouts_data = []
    for elite in elites:
        binds = {"base": {}, "shift": {}, "alt": {}, "shift_alt": {}}
        for idx, kbi in enumerate(elite[2]):
            if kbi is None:
                continue
            if idx < layout.sizes[0]: layer = "base"
            elif idx < layout.sizes[0] + layout.sizes[1]: layer = "shift"
            elif len(layout.sizes) > 2 and idx < layout.sizes[0] + layout.sizes[1] + layout.sizes[2]: layer = "alt"
            else: layer = "shift_alt"
            binds[layer][layout.idx2key[idx]] = layout.keybinds[kbi]
        best_layouts_data.append({
            "score": elite[0],
            "metrics": dict(zip(objectives, elite[1])),
            "binds": binds
        })
    try:
        with open("output/best_layouts.json", "w", encoding="utf-8") as f:
            json.dump(best_layouts_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"error saving best layouts: {e}")

    if paras.dev_mode:
        print(f"SWAP ACCEPTANCE RATE:  {total_swap_accept / paras.generation_limit * n_swaps / 2}")
        print(f"JUMP ACCEPTANCE RATE:  {total_jump_accept / (paras.generation_limit - total_better_jumps)}")
        print(f"BETTER JUMPS RATE:  {total_better_jumps / paras.generation_limit}")


def save_xkb_file(layout, elite, filename):
    # Gather binds
    binds = {"base": {}, "shift": {}, "alt": {}, "shift_alt": {}}
    for idx, kbi in enumerate(elite[2]):
        if kbi is None:
            continue
        if idx < layout.sizes[0]: layer = "base"
        elif idx < layout.sizes[0] + layout.sizes[1]: layer = "shift"
        elif len(layout.sizes) > 2 and idx < layout.sizes[0] + layout.sizes[1] + layout.sizes[2]: layer = "alt"
        else: layer = "shift_alt"
        binds[layer][layout.idx2key[idx]] = layout.keybinds[kbi]

    # Map layout.txt key names to standard XKB keycodes
    layout_to_xkb = {
        "`": "TLDE", "1": "AE01", "2": "AE02", "3": "AE03", "4": "AE04", "5": "AE05",
        "6": "AE06", "7": "AE07", "8": "AE08", "9": "AE09", "0": "AE10", "-": "AE11", "=": "AE12",
        "tab": "TAB", "q": "AD01", "w": "AD02", "e": "AD03", "r": "AD04", "t": "AD05",
        "y": "AD06", "u": "AD07", "i": "AD08", "o": "AD09", "p": "AD10", "[": "AD11", "]": "AD12",
        "caps": "CAPS", "a": "AC01", "s": "AC02", "d": "AC03", "f": "AC04", "g": "AC05",
        "h": "AC06", "j": "AC07", "k": "AC08", "l": "AC09", ";": "AC10", "'": "AC11", "\\": "BKSL",
        "mback": "AC11",
        "lsft": "LFSH", "z": "AB01", "x": "AB02", "c": "AB03", "v": "AB04", "b": "AB05",
        "n": "AB06", "m": "AB07", ",": "AB08", ".": "AB09", "/": "AB10", "rsft": "RTSH",
        "lctl": "LCTL", "lmet": "LWIN", "lalt": "LALT", "spc": "SPCE", "ralt": "RALT", "rctl": "RCTL"
    }

    # Helper to convert character / bind name to XKB token
    def to_xkb_sym(char):
        if char is None:
            return "U0000"
        if char in ("spc", "space", "_"):
            return "space"
        if char in ("bspc", "backspace", "<"):
            return "BackSpace"
        if char in ("home",):
            return "Home"
        if char in ("lsft", "rsft", "shift"):
            return "Shift_L"
        if char in ("lalt", "ralt", "alt"):
            return "ISO_Level3_Shift"
        if char in ("lctl",):
            return "Control_L"
        if char in ("lmet",):
            return "Super_L"
        
        # If it's a single character
        if len(char) == 1:
            return f"U{ord(char):04X}"
        
        return "U0000"

    lines = []
    lines.append("partial alphanumeric_keys")
    lines.append('xkb_symbols "layout" {')
    lines.append('    name[Group1] = "layout search crafting.";')
    lines.append("")
    lines.append('    key <RALT> {[  ISO_Level3_Shift  ], type[group1]="ONE_LEVEL" };')

    for phys_key in sorted(layout.keys.keys()):
        if phys_key not in layout_to_xkb:
            continue
        xkb_code = layout_to_xkb[phys_key]
        if xkb_code == "RALT":
            continue
        
        c1 = binds["base"].get(phys_key)
        c2 = binds["shift"].get(phys_key)
        c3 = binds["alt"].get(phys_key)
        c4 = binds["shift_alt"].get(phys_key)

        if c1 is None and c2 is None and c3 is None and c4 is None:
            continue
        
        # Auto-uppercase for shift layer if not specified
        if c2 is None and c1 is not None and len(c1) == 1:
            c2 = c1.upper()
        elif c2 is not None and len(c2) == 1:
            c2 = c2.upper()

        sym1 = to_xkb_sym(c1)
        sym2 = to_xkb_sym(c2)
        sym3 = to_xkb_sym(c3)
        sym4 = to_xkb_sym(c4)

        lines.append(f"    key <{xkb_code}> {{ [ {sym1}, {sym2}, {sym3}, {sym4} ] }};")

    lines.append("};")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\tsaved xkb layout: {filename}")
    except Exception as e:
        print(f"error generating xkb file: {e}")



if __name__ == '__main__':
    run()