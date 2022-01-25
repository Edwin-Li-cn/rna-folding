from os.path import dirname, join
from collections import defaultdict
from itertools import product, combinations
import os
import click
import matplotlib
import numpy as np
import networkx as nx
import dimod
from dwave.system import LeapHybridCQMSampler
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
#%%
base_dir = os.getcwd()
DEFAULT_PATH = join(base_dir, 'RNA_text_files', 'TMGMV_UPD-PK1.txt')
with open(DEFAULT_PATH) as f:
    rna = "".join(("".join(line.split()[1:]) for line in f.readlines())).lower()
#%%
index_dict = defaultdict(list)
for i, nucleotide in enumerate(rna):
    index_dict[nucleotide].append(i)
#%%
hydrogen_bonds = [('a', 't'), ('a', 'u'), ('c', 'g'), ('g', 't'), ('g', 'u')]
bond_matrix = np.zeros((len(rna), len(rna)), dtype=bool)
for pair in hydrogen_bonds:
    for bond in product(index_dict[pair[0]], index_dict[pair[1]]):
        if abs(bond[0] - bond[1]) > 2:
            bond_matrix[min(bond), max(bond)] = 1

#%%
min_stem = 3
min_loop = 2
stem_dict = {}
n = bond_matrix.shape[0]
# Iterate through matrix looking for possible stems.
for i in range(n + 1 - (2 * min_stem + min_loop)):
    for j in range(i + 2 * min_stem + min_loop - 1, n):
        if bond_matrix[i, j]:
            k = 1
            # Check down and left for length of stem.
            # Note that bond_matrix is strictly upper triangular, so loop will terminate.
            while bond_matrix[i + k, j - k]:
                bond_matrix[i + k, j - k] = False
                k += 1

            if k >= min_stem:
                # A 4-tuple is used to represent the stem.
                stem_dict[(i, i + k - 1, j - k + 1, j)] = []

# Iterate through all sub-stems weakly contained in a maximal stem under inclusion.
for stem in stem_dict.keys():
    stem_dict[stem].extend([(stem[0] + i, stem[0] + k, stem[3] - k, stem[3] - i)
                            for i in range(stem[1] - stem[0] - min_stem + 2)
                            for k in range(i + min_stem - 1, stem[1] - stem[0] + 1)])
    #%%
    c=0.3
pseudos = {}
# Look within all pairs of maximal stems for possible pseudoknots.
for stem1, stem2 in product(stem_dict.keys(), stem_dict.keys()):
    # Using product instead of combinations allows for short asymmetric checks.
    if stem1[0] + 2 * min_stem < stem2[1] and stem1[2] + 2 * min_stem < stem2[3]:
        pseudos.update({(substem1, substem2): c * (1 + substem1[1] - substem1[0]) * (1 + substem2[1] - substem2[0])
                        for substem1, substem2
                        in product(stem_dict[stem1], stem_dict[stem2])
                        if substem1[1] < substem2[0] and substem2[1] < substem1[2] and substem1[3] < substem2[2]})
#%%



#%%
cqm = dimod.ConstrainedQuadraticModel()
for stem, substems in stem_dict.items():
    if len(substems) > 1:
        # Add the variable for all zeros case in one-hot constraint
        zeros = 'Null:' + str(stem)
        cqm.add_variable(zeros, 'BINARY')
        cqm.add_discrete(substems + [zeros], stem)