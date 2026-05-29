import random
import math
from classes import Layout
import bisect

class Mutator:

    def __init__(self, layout: Layout):
        self.layout = layout
        self.max_attempts = 10
        self.n = len(layout.idx2key)
        self.fixed_keys = layout.fixed_keys
        self.special_set = layout.available_skb_set
        self.base_end = layout.sizes[0]
        self.valid_indices = layout.available_keys

    def CASH_pick(self):
        r = random.random() * self.layout.cum_avai_strain_heapmap[-1]
        return self.layout.available_keys[bisect.bisect_left(self.layout.cum_avai_strain_heapmap, r)]

    def binary_swap(self, ind: list, threshold: float):
        new_ind = ind[:]

        if len(self.valid_indices) < 2:
            return new_ind

        special_cps = {}
        special_cps_inverse = {}
        for kbi in range(self.base_end):
            if new_ind[kbi] in self.special_set:
                cps = self.layout.counterparts[kbi]
                special_cps[kbi] = cps
                for cp in cps:
                    special_cps_inverse[cp] = kbi

        for i in self.valid_indices:
            if random.random() > threshold:
                continue
            kbi = new_ind[i]
            for _ in range(self.max_attempts):
                j = self.CASH_pick()
                if i == j:
                    continue

                kbj = new_ind[j]
                if kbi == kbj == None:
                    continue

                ok = True
                si = False
                sj = False

                if i in special_cps_inverse or j in special_cps_inverse:
                    ok = False

                cpj = self.layout.counterparts[j]
                cpi = self.layout.counterparts[i]

                if i < self.base_end and kbi in self.special_set:
                    if j >= self.base_end or (j < self.base_end and any(cp < len(new_ind) and new_ind[cp] is not None for cp in cpj)):
                        ok = False
                    elif i in special_cps:
                        for cp in special_cps[i]:
                            special_cps_inverse.pop(cp, None)
                        special_cps.pop(i)
                    sj = True
                if j < self.base_end and kbj in self.special_set:
                    if i >= self.base_end or (i < self.base_end and any(cp < len(new_ind) and new_ind[cp] is not None for cp in cpi)):
                        ok = False
                    elif j in special_cps:
                        for cp in special_cps[j]:
                            special_cps_inverse.pop(cp, None)
                        special_cps.pop(j)
                    si = True

                if not ok:
                    continue

                new_ind[i], new_ind[j] = kbj, kbi
                if si and cpi:
                    special_cps[i] = cpi
                    for cp in cpi:
                        special_cps_inverse[cp] = i
                if sj and cpj:
                    special_cps[j] = cpj
                    for cp in cpj:
                        special_cps_inverse[cp] = j

                break

        return new_ind

    def layer_swap(self, ind: list, threshold: float):
        new_ind = ind[:]
        for i in range(self.base_end):
            cps = self.layout.counterparts.get(i, [])
            if not cps or random.random() > threshold or i in self.fixed_keys or new_ind[i] in self.special_set:
                continue

            cp = random.choice(cps)
            if cp in self.fixed_keys:
                continue
            new_ind[i], new_ind[cp] = new_ind[cp], new_ind[i]
        return new_ind

    def mutate(self, ind, T):
        threshold = max(0.1 * 0.3 + T * 0.7, 1.0 / len(ind))
        if random.random() < 0.9:
            ind = self.binary_swap(ind, threshold)
        else:
            ind = self.layer_swap(ind, threshold)
        return ind


import numpy as np
def swap_replicas(temperatures, replicas, evaluations, scores, generation_count):
    start_idx = generation_count % 2
    accept_count = np.zeros(len(replicas) - 1)
    for i in range(start_idx, len(replicas) - 1, 2):
        delta_beta = 1.0 / temperatures[i] - 1.0 / temperatures[i + 1]
        delta_score = scores[i] - scores[i + 1]
        arg = max(-700.0, min(700.0, delta_beta * delta_score))
        if random.random() < min(1, math.exp(arg)):
            accept_count[i] += 1
            replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
            evaluations[i], evaluations[i + 1] = evaluations[i + 1], evaluations[i]
            scores[i], scores[i + 1] = scores[i + 1], scores[i]

    return replicas, evaluations, scores, accept_count