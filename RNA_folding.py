# Copyright 2021 D-Wave Systems
# Based on the paper 'RNA folding using quantum computers’
# Fox DM, MacDermaid CM, Schreij AM, Zwierzyna M, Walker RC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import dirname, join
from collections import defaultdict
from itertools import product, combinations

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


def text_to_matrix(file_name, min_loop):
    """ Reads properly formatted RNA text file and returns a matrix of possible hydrogen bonding pairs.

    Args:
        file_name (str):
            Path to text file.
        min_loop (int):
            Minimum number of nucleotides separating two sides of a stem.

    Returns:
        :class: `numpy.ndarray`:
            Numpy matrix of 0's and 1's, where 1 represents a possible bonding pair.
    """

    # Requires text file of RNA data written in same format as examples.
    with open(file_name) as f:
        rna = "".join(("".join(line.split()[1:]) for line in f.readlines())).lower()
    #读取文件的rna序列，'aaagucgcugaagacuuaaaauucagg'即这种形式

    # Create a dictionary of all indices where each nucleotide occurs.
    index_dict = defaultdict(list)

    # Create a dictionary giving list of indices for each nucleotide.
    for i, nucleotide in enumerate(rna):
        index_dict[nucleotide].append(i)
    # defaultdict(<class 'list'>, {'a': [0, 1, 2], 'g': [3, 6], 'u': [4], 'c': [5, 7]})即这种形式

    # List of possible hydrogen bonds for stems.
    # Recall that 't' is sometimes used as a stand-in for 'u'.
    hydrogen_bonds = [('a', 't'), ('a', 'u'), ('c', 'g'), ('g', 't'), ('g', 'u')]
    #可能形成氢键的组合

    # Create a upper triangular 0/1 matrix indicating where bonds may occur.
    bond_matrix = np.zeros((len(rna), len(rna)), dtype=bool)
    for pair in hydrogen_bonds:
        for bond in product(index_dict[pair[0]], index_dict[pair[1]]):
            if abs(bond[0] - bond[1]) > min_loop:
                bond_matrix[min(bond), max(bond)] = 1
    # 上半三角矩阵构建，pair为可能形成氢键的组合的依次迭代取值('a', 't'), ('a', 'u'), ('c', 'g'), ('g', 't'), ('g', 'u')；如pair = ('a', 'u')
    # index_dict[pair[0]]=index_dict['a']=[0, 1, 2]；index_dict[pair[1]]=index_dict['u']=[4]
    # bound在product循环，例如=(0, 4)，bound[0]=0；bound[1]=4，判断可以形成碱基对的核苷酸位置是否大于min_loop，大于就把相应bond_matrix位置赋值1

    return bond_matrix


def make_stem_dict(bond_matrix, min_stem, min_loop):
    """ Takes a matrix of potential hydrogen binding pairs and returns a dictionary of possible stems.

    The stem dictionary records the maximal stems (under inclusion) as keys,
    where each key maps to a list of the associated stems weakly contained within the maximal stem.
    Recording stems in this manner allows for faster computations.

    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Numpy matrix of 0's and 1's, where 1 represents a possible bonding pair.
        min_stem (int):
            Minimum number of nucleotides in each side of a stem.
        min_loop (int):
            Minimum number of nucleotides separating two sides of a stem.

    Returns:
        dict: Dictionary of all possible stems with maximal stems as keys.
    """

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

    # 同An efficient伪代码，i,j范围容易判断，考虑极限情况即可，将i,j间最长的stem记录在stem_dict的key里，value设为空列表
    # 比如stem_dict: {(1, 5, 12, 16): [], (2, 4, 20, 22): [], (7, 11, 21, 25): [], (8, 10, 15, 17): [], (14, 16, 24, 26): []}

    # Iterate through all sub-stems weakly contained in a maximal stem under inclusion.
    for stem in stem_dict.keys():
        stem_dict[stem].extend([(stem[0] + i, stem[0] + k, stem[3] - k, stem[3] - i)
                                for i in range(stem[1] - stem[0] - min_stem + 2)
                                for k in range(i + min_stem - 1, stem[1] - stem[0] + 1)])

    # 将stem_dict里每个key值的stem的子stem添加到stem_dict里相应value列表中去：(1, 5, 12, 16):[(1, 3, 14, 16), (1, 4, 13, 16), (1, 5, 12, 16), (2, 4, 13, 15), (2, 5, 12, 15), (3, 5, 12, 14)]
    # 理解：固定i看k：k最小要和stem[0] + i位置间隔min_stem - 1；k最大要为给定stem的最长长度stem[1] - stem[0] + 1)
    #      i最大使得stem[0]和stem[1]间隔min_stem - 1
    # {(1, 5, 12, 16): [(1, 3, 14, 16), (1, 4, 13, 16), (1, 5, 12, 16), (2, 4, 13, 15), (2, 5, 12, 15), (3, 5, 12, 14)], (2, 4, 20, 22): [(2, 4, 20, 22)], (7, 11, 21, 25): [(7, 9, 23, 25), (7, 10, 22, 25), (7, 11, 21, 25), (8, 10, 22, 24), (8, 11, 21, 24), (9, 11, 21, 23)], (8, 10, 15, 17): [(8, 10, 15, 17)], (14, 16, 24, 26): [(14, 16, 24, 26)]}
    return stem_dict


def check_overlap(stem1, stem2):
    """ Checks if 2 stems use any of the same nucleotides.

    Args:
        stem1 (tuple):
            4-tuple containing stem information.
        stem2 (tuple):
            4-tuple containing stem information.

    Returns:
         bool: Boolean indicating if the two stems overlap.
    """

    # Check for string dummy variable used when implementing a discrete variable.
    if type(stem1) == str or type(stem2) == str:
        return False
    #避免bug

    # Check if any endpoints of stem2 overlap with stem1.
    for val in stem2:
        if stem1[0] <= val <= stem1[1] or stem1[2] <= val <= stem1[3]:
            return True
    # Check if endpoints of stem1 overlap with stem2.
    # Do not need to check all stem1 endpoints.
    for val in stem1[1:3]:
        if stem2[0] <= val <= stem2[1] or stem2[2] <= val <= stem2[3]:
            return True

    return False


def pseudoknot_terms(stem_dict, min_stem=3, c=0.3):
    """ Creates a dictionary with all possible pseudoknots as keys and appropriate penalties as values.

    The penalty is the parameter c times the product of the lengths of the two stems in the knot.

    Args:
        stem_dict (dict):
            Dictionary with maximal stems as keys and list of weakly contained sub-stems as values.
        min_stem (int):
            Smallest number of consecutive bonds to be considered a stem.
        c (float):
            Parameter factor of the penalty on pseudoknots.

    Returns:
         dict: Dictionary with all possible pseudoknots as keys and appropriate penalty as as value pair.
    """

    pseudos = {}
    # Look within all pairs of maximal stems for possible pseudoknots.
    for stem1, stem2 in product(stem_dict.keys(), stem_dict.keys()):
        # 先找保存在stem_dict的key里最大的stem
        # Using product instead of combinations allows for short asymmetric checks.
        if stem1[0] + 2 * min_stem < stem2[1] and stem1[2] + 2 * min_stem < stem2[3]:
            # 这个判断条件存疑？
            # 当stem2:(a',b',c',d')和stem1:(a,b,c,d);有可能交叉(aba'b'cdc'd')(间隔min_stem)再进行下一步判断：
            pseudos.update({(substem1, substem2): c * (1 + substem1[1] - substem1[0]) * (1 + substem2[1] - substem2[0])
                            for substem1, substem2
                            in product(stem_dict[stem1], stem_dict[stem2])
                            if substem1[1] < substem2[0] and substem2[1] < substem1[2] and substem1[3] < substem2[2]})
            # 进一步判断保存在stem_dict的value里小的stem，同样交叉aba'b'cdc'd'，若pseuknot形成，添加penallty系数到pseudos字典key(substem1, substem2)的value
    return pseudos


def make_plot(file, stems, fig_name='RNA_plot'):
    """ Produces graph plot and saves as .png file.

    Args:
        file (str):
            Path to text file name containing RNA information.
        stems (list):
            List of stems in solution, encoded as 4-tuples.
        fig_name (str):
            Name of file created to save figure. ".png" is added automatically
    """

    # Read RNA file for length and labels.
    with open(file) as f:
        rna = "".join(("".join(line.split()[1:]) for line in f.readlines())).lower()
    #读取rna序列
    # Create graph with edges from RNA sequence and stems. Nodes are temporarily labeled by integers.
    G = nx.Graph()
    rna_edges = [(i, i + 1) for i in range(len(rna) - 1)]
    #rna序列连线
    stem_edges = [(stem[0] + i, stem[3] - i) for stem in stems for i in range(stem[1] - stem[0] + 1)]
    #bps连线
    G.add_edges_from(rna_edges + stem_edges)

    # Assign each nucleotide to a color.
    color_map = []
    for node in rna:
        if node == 'g':
            color_map.append('tab:red')
        elif node == 'c':
            color_map.append('tab:green')
        elif node == 'a':
            color_map.append('y')
        else:
            color_map.append('tab:blue')
    #不同核苷酸画不同颜色

    options = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.8}
    pos = nx.spring_layout(G, iterations=5000)  # max(3000, 125 * len(rna)))
    #用Fruchterman-Reingold算法排列节点
    nx.draw_networkx_nodes(G, pos, node_color=color_map, **options)
    #画节点

    labels = {i: rna[i].upper() for i in range(len(rna))}
    #labels字典，字母转化为大写
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="whitesmoke")
    #把RNA序列字母AUCGT画出

    nx.draw_networkx_edges(G, pos, edgelist=rna_edges, width=3.0, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=stem_edges, width=4.5, alpha=0.7, edge_color='tab:pink')
    #画边

    plt.savefig(fig_name + '.png')

    print('\nPlot of solution saved as {}.png'.format(fig_name))


def build_cqm(stem_dict, min_stem, c):
    """ Creates a constrained quadratic model to optimize most likely stems from a dictionary of possible stems.

    Args:
        stem_dict (dict):
            Dictionary with maximal stems as keys and list of weakly contained sub-stems as values.
        min_stem (int):
            Smallest number of consecutive bonds to be considered a stem.
        c (float):
            Parameter factor of the penalty on pseudoknots.

    Returns:
         :class:`~dimod.ConstrainedQuadraticModel`: Optimization model for RNA folding.
    """

    # Create linear coefficients of -k^2, prioritizing inclusion of long stems.
    # We depart from the reference paper in this formulation.
    linear_coeffs = {stem: -1 * (stem[1] - stem[0] + 1) ** 2 for sublist in stem_dict.values() for stem in sublist}
    #线性项系数k方的构造，把每个可能的stem都考虑进去

    # Create constraints for overlapping and and sub-stem containment.
    quadratic_coeffs = pseudoknot_terms(stem_dict, min_stem=min_stem, c=c)
    #返回字典，key值为所有可能的pesudoknot项，value值为相应的penalty系数
    bqm = dimod.BinaryQuadraticModel(linear_coeffs, quadratic_coeffs, 'BINARY')
    #二进制二次模型调用
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(bqm)
    #CQM求解bqm模型

    # Add constraint disallowing overlapping sub-stems included in same maximal stem.
    for stem, substems in stem_dict.items():
        if len(substems) > 1:
            # Add the variable for all zeros case in one-hot constraint
            zeros = 'Null:' + str(stem)
            cqm.add_variable(zeros, 'BINARY')
            cqm.add_discrete(substems + [zeros], stem)
            #添加变量限制，包括最长的stem在内，子stem只能取1个

    for stem1, stem2 in combinations(stem_dict.keys(), 2):
        # Check maximal stems first.
        if check_overlap(stem1, stem2):
            # If maximal stems overlap, compare list of smaller stems.
            for stem_pair in product(stem_dict[stem1], stem_dict[stem2]):
                if check_overlap(stem_pair[0], stem_pair[1]):
                    cqm.add_constraint(dimod.quicksum([dimod.Binary(stem) for stem in stem_pair]) <= 1)
                    #添加overlap_constraint，存疑？

    return cqm


def process_cqm_solution(sample_set, verbose=True):
    """ Processes samples from solution and prints relevant information.

    Prints information about the best feasible solution and returns a list of stems contained in solution.
    Returns solution as a list of stems rather than a binary string.

    Args:
        sample_set:
            :class:`~dimod.SampleSet`: Sample set of formed by sampling the RNA folding optimization model.
        verbose (bool):
            Boolean indicating if function should print additional information.

    Returns:
        list: List of stems included in optimal solution, encoded as 4-tuples.
    """

    # Filter for feasibility.
    feasible_samples = sample_set.filter(lambda s: s.is_feasible)
    #检查可行解
    # Check that feasible example exists.
    if not feasible_samples:
        raise Exception("All solutions infeasible. You may need to try again.")

    # Extract best feasible sample.
    solution = feasible_samples.first

    print('Best Energy:', solution.energy)

    # Extract stems with a positive indicator variable.
    bonded_stems = [stem for stem, val in solution.sample.items() if val == 1 and type(stem) == tuple]
    #取出stems，存到bonded_stems列表中
    print('\nNumber of stems in best solution:', len(bonded_stems))
    print('Stems in best solution:', *bonded_stems)

    if verbose:
        print('\nNumber of variables (stems):', len(solution[0].keys()))

        # Find pseudoknots using product instead of combinations allows for short asymmetric checks.
        pseudoknots = [(stem1, stem2) for [stem1, stem2] in product(bonded_stems, bonded_stems)
                       if stem1[1] < stem2[0] and stem2[1] < stem1[2] and stem1[3] < stem2[2]]
        #检测pseudoknot，存到列表pseudoknots中
        print('\nNumber of pseudoknots in best solution:', len(pseudoknots))
        if pseudoknots:
            print('Pseudoknots:', *pseudoknots)

    return bonded_stems


# Create command line functionality.
DEFAULT_PATH = join(dirname(__file__), 'RNA_text_files', 'TMGMV_UPD-PK1.txt')


@click.command(help='Solve an instance of the RNA folding problem using '
                    'LeapHybridCQMSampler.')
@click.option('--path', type=click.Path(), default=DEFAULT_PATH,
              help=f'Path to problem file.  Default is {DEFAULT_PATH!r}')
@click.option('--verbose', default=True,
              help='Prints additional model information.')
@click.option('--min-stem', type=click.IntRange(1,), default=3,
              help='Minimum length for a stem to be considered.')
@click.option('--min-loop', type=click.IntRange(0,), default=2,
              help='Minimum number of nucleotides separating two sides of a stem.')
@click.option('-c', type=click.FloatRange(0,), default=0.3,
              help='Multiplier for the coefficient of the quadratic terms for pseudoknots.')
def main(path, verbose, min_stem, min_loop, c):
    """ Find optimal stem configuration of an RNA sequence.

    Reads file, creates constrained quadratic model, solves model, and creates a plot of the result.
    Default parameters are set by click module inputs.

    Args:
        path (str):
            Path to problem file with RNA sequence.
        verbose (bool):
            Boolean to determine amount of information printed.
        min_stem (int):
            Smallest number of consecutive bonds to be considered a stem.
        min_loop (int):
            Minimum number of nucleotides separating two sides of a stem.
        c (float):
            Multiplier for the coefficient of the quadratic terms for pseudoknots.

    Returns:
        None: None
    """
    if verbose:
        print('\nPreprocessing data from:', path)
    #verbose判断是否详细输出处理过程参数

    matrix = text_to_matrix(path, min_loop)
    #将txtRNA序列文本文件转化为上半三角矩阵，矩阵元素为一代表对应位置核苷酸形成碱基对
    stem_dict = make_stem_dict(matrix, min_stem, min_loop)
    #将矩阵转化为stem_dict，key是最长的stem，value是相应stem的子stem
    if stem_dict:
        cqm = build_cqm(stem_dict, min_stem, c)
    #建立constrained quadratic model：cqm
    else:
        print('\nNo possible stems were found. You may need to check your parameters.')
        return None

    if verbose:
        print('Connecting to Solver...')

    sampler = LeapHybridCQMSampler()

    if verbose:
        print('Finding Solution...')

    sample_set = sampler.sample_cqm(cqm)
    sample_set.resolve()
    #求解器求解

    if verbose:
        print('Processing solution...')

    stems = process_cqm_solution(sample_set, verbose)
    #二进制解化为stem(a,b,c,d)形式
    make_plot(path, stems)
    #解可视化作图


if __name__ == "__main__":
    main()
