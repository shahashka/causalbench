"""
Copyright (C) 2022  GlaxoSmithKline plc - Mathieu Chevalley;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import random
from typing import List, Any
import pickle

import numpy as np
import pandas as pd
import scprep
import networkx as nx
from causallearn.graph import GeneralGraph
from numpy.random import RandomState
import scipy
import itertools
from sklearn.metrics import mutual_info_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
    
def load_random_state(
    random_state: RandomState | int | None = None,
) -> RandomState:
    if random_state is None:
        return RandomState()
    elif isinstance(random_state, RandomState):
        return random_state
    elif isinstance(random_state, int):
        return RandomState(random_state)
    else:
        raise ValueError(
            "Illegal value for `load_random_state()` Must be either an "
            "instance of `RandomState`, an integer to seed with, or None."
        )
        
def causallearn_graph_to_edges(G: GeneralGraph, gene_names: List[str]):
    node_map = G.get_node_map()
    edges = []
    for edge in G.get_graph_edges():
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        if gene_names is None:
            edges.append((node1.get_name(), node2.get_name()))
        else:
            node1_id = node_map[node1]
            node2_id = node_map[node2]
            edges.append((gene_names[node1_id], gene_names[node2_id]))
    return edges

def partion_network(gene_names, partitions_length, seed):
    random.seed(seed)
    indices = list(range(len(gene_names)))
    random.shuffle(indices)
    partition_length = int(len(indices) / partitions_length)
    partitions = [indices[i::partition_length] for i in range(partition_length)]
    return partitions
from collections import deque

def modularity_partition(
    superstructure: np.ndarray,
    gene_names:List[Any],
    resolution: int = 1,
    cutoff: int = 1,
    best_n: int = None,
):
    """
    Creates disjoint partition by greedily maximizing modularity. Using
    networkx built-in implementaiton.

    Args:
        superstructure (np.ndarray): the adjacency matrix for the superstructure
        resolution (float): resolution parameter, trading off intra- versus
            inter-group edges.
        cutoff (int): lower limit on number of communities before termination
        best_n (int): upper limit on number of communities before termination
        See networkx documentation for more.

    Returns:
        dict: the estimated partition as a dictionary {comm_id : [nodes]}
    """
    G = nx.from_numpy_array(superstructure)
    G = nx.relabel_nodes(G, dict(zip(np.arange(len(gene_names)), gene_names)))
    community_lists = nx.community.greedy_modularity_communities(
        G, resolution=resolution, cutoff=cutoff, best_n=best_n
    )

    partition = dict()
    for idx, c in enumerate(community_lists):
        partition[idx] = list(c)
    return partition

def rand_edge_cover_partition(
    superstructure: np.ndarray,
    gene_names: List[Any],
    resolution: int = 1,
    cutoff: int = 1,
    best_n: int = None,
):
    """Creates a random edge covering partition.

    Uses greedy modularity to create a disjoint partition. Then, randomly
    chooses cut edges and randomly assigns endpoints to communities.
    Recursively adds any shared endpoints to the same community

    Args:
        adj_mat (np.ndarray): Adjacency matrix for the graph
        data (pd.DataFrame): unused parameter
        resolution (float): resolution parameter, trading off intra-
            versus inter-group edges.
        cutoff (int): lower limit on number of communities before termination
        best_n (int): upper limit on number of communities before termination

    Returns:
        dict: the overlapping partition as a dictionary {comm_id : [nodes]}
    """
    #partition = bounded_graph_clustering(superstructure, max_cluster_size=30)
    partition = modularity_partition(superstructure, gene_names=gene_names, resolution=resolution, best_n=best_n, cutoff=cutoff)
    #compressed_partition = dict()
    print(f"Number of initial partitions: {len(partition.keys())}, biggest: {max([len(v) for v in partition.values()])}, smallest {min([len(v) for v in partition.values()])}")
    print([len(v) for v in partition.values()])

    G = nx.from_numpy_array(superstructure)
    G = nx.relabel_nodes(G, dict(zip(np.arange(len(gene_names)), gene_names)))
    # degrees = [G.degree(v) for v in G.nodes()]
    # print(f"Max degree in graph is {max(degrees)}")
    def edge_coverage_helper(i, j, comm, cut_edges, node_to_comm):
        if comm not in node_to_comm[i]:
            node_to_comm[i] += [comm]
        if comm not in node_to_comm[j]:
            node_to_comm[j] += [comm]
        cut_edges.remove((i, j))
        return node_to_comm, cut_edges

    node_to_comm = dict()
    for comm_id, comm in partition.items():
        for node in comm:
            node_to_comm[node] = [comm_id]
    cut_edges = []
    for edge in G.edges():
        if node_to_comm[edge[0]] != node_to_comm[edge[1]]:
            cut_edges.append(edge)
    # Randomly choose a cut edge until all edges are covered
    while len(cut_edges) > 0:
        edge_ind = np.random.choice(np.arange(len(cut_edges)))
        i = cut_edges[edge_ind][0]
        j = cut_edges[edge_ind][1]

        # Randomly choose an endpoint and associated community
        possible_comms = list(set(node_to_comm[i] + node_to_comm[j]))
        comm = np.random.choice(possible_comms)
        node_to_comm, cut_edges = edge_coverage_helper(
            i, j, comm, cut_edges, node_to_comm
        )

    edge_cover_partition = dict()
    # Update the disjoint partition
    for n, comms in node_to_comm.items():
        for c in comms:
            if c in edge_cover_partition.keys():
                edge_cover_partition[c] += [n]
            else:
                edge_cover_partition[c] = [n]
    sizes = [len(v) for v in edge_cover_partition.values()]
    print(f"Number of partitions: {len(edge_cover_partition.keys())}, biggest: {max(sizes)}, smallest {min(sizes)}")
    return edge_cover_partition

def expansive_causal_partition(
    superstructure: np.ndarray,
    gene_names: List[Any],
    resolution: int = 1,
    cutoff: int = 1,
    best_n: int = None,
):
    """
    Creates a causal partition by adding the outer-boundary of each cluster
    to that cluster.

    First uses greedy modularity to create a disjoint partition, then adds
    the outer-boundary of each cluster to create a causal partition

    Args:
        superstructure (np.ndarray): the adjacency matrix for the superstructure
        resolution (float): resolution parameter, trading off intra- versus
            inter-group edges.
        cutoff (int): lower limit on number of communities before termination
        best_n (int): upper limit on number of communities before termination

    Returns:
        dict: the causal partition as a dictionary {comm_id : [nodes]}
    """
    partition = modularity_partition(superstructure, gene_names=gene_names, resolution=resolution, best_n=best_n, cutoff=cutoff)
    #partition = bounded_graph_clustering(superstructure, max_cluster_size=30)
    print(f"Number of initial partitions: {len(partition.keys())}, biggest: {max([len(v) for v in partition.values()])}, smallest {min([len(v) for v in partition.values()])}")

    G = nx.from_numpy_array(superstructure)
    G = nx.relabel_nodes(G, dict(zip(np.arange(len(gene_names)), gene_names)))

    causal_partition = dict()
    for idx, c in enumerate(list(partition.values())):
        outer_node_boundary = nx.node_boundary(G, c)
        expanded_cluster = set(c).union(outer_node_boundary)
        causal_partition[idx] = list(expanded_cluster)
    sizes = [len(v) for v in causal_partition.values()]
    print(f"Partition sizes: {sizes}")
    return causal_partition
    
# def artificial_superstructure(
#     eval_network: np.ndarray,
#     frac_retain_direction: float = 0.1,
#     frac_extraneous: float = 0.5,
# ) -> np.ndarray:
#     """
#     Creates a superstructure by discarding some of the directions in edges of
#     G_star and adding extraneous edges.

#     Args:
#         G_star_adj_mat (np.ndarray): the adjacency matrix for the target graph
#         frac_retain_direction (float): what percentage of edges will retain
#             their direction information
#         frac_extraneous (float): adds frac_extraneous*m many additional
#             edges, for m the number of edges in G_star

#     Returns:
#         An adjacency matrix for the superstructure we've created
#     """
#     def pick_k_random_edges(k, nodes):
#         return list(zip(random.choices(nodes, k=k), random.choices(nodes, k=k)))

#     G_star = nx.from_numpy_array(eval_network, create_using=nx.DiGraph())

#     # returns a deepcopy
#     G_super = G_star.to_undirected()
#     # add extraneous edges
#     m = G_star.number_of_edges()
#     nodes = list(G_star.nodes())
#     G_super.add_edges_from(
#         pick_k_random_edges(k=int(frac_extraneous * m), nodes=nodes)
#     )

#     return nx.adjacency_matrix(G_super).toarray()

# def calc_MI(x, y, bins):
#     c_xy = np.histogram2d(x, y, bins)[0]
#     mi = mutual_info_score(None, None, contingency=c_xy)
#     return mi
# def load_ppi_superstructure(ordered_genes) -> np.array:
#     num_genes = len(ordered_genes)
#     ordered_genes = list(ordered_genes)
#     data_directory = "./causalscbench/data_access/data"
#     dataset_name = "weissmann_k562" # hardcode for now
#     (
#             corum,
#             lr_pairs,
#             string_network_pairs,
#             string_physical_pairs,
#             chipseq_pairs,
#         ) = CreateEvaluationDatasets(data_directory, dataset_name).load()
#     ss = np.zeros((num_genes, num_genes))
#     print(len(string_physical_pairs))
#     for edge in string_physical_pairs:
#         if edge[0] in ordered_genes and edge[1] in ordered_genes:
#             i = ordered_genes.index(edge[0])
#             j = ordered_genes.index(edge[1])
#             ss[i,j] = 1
#     return ss
    

def _permute_and_max_corr(args):
    matrix, seed = args
    rng = np.random.default_rng(seed)

    # shuffle each row independently
    shuffled = np.apply_along_axis(rng.permutation, 1, matrix)

    corr = np.corrcoef(shuffled, rowvar=False)
    np.fill_diagonal(corr, 0)

    return np.max(corr)


def correlation_superstructure(
    expression_matrix: np.ndarray, 
    seed: int, 
    num_iterations: int = 200
) -> np.ndarray:
    """
    Creates a superstructure by calculating the correlation matrix from
    the data.

    A cutoff value is chosen using permutation testing: randomly shuffling the
    data amtrix and recalculating the correlation matrix over a specified
    number of iterations. The upper bound of the 95% confidence interval
    for the maximum value in each shuffled matrix is used as the threshold
    for the superstructure.

    Args:
        expression_matrix (np.ndarray) : sampled data set, each column is a random
            variable
        num_iterations (int) : number of iterations for permutation testing


    Returns:
        an adjacency matrix for the superstructure we've created
    """
    expression_matrix = pd.DataFrame(data=expression_matrix)
    corr_mat = expression_matrix.corr("pearson").to_numpy()
    np.fill_diagonal(corr_mat, 0)
    random_corr_coef = []
    # prepare seeds for reproducibility
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10**9, size=num_iterations)
    tasks = [(expression_matrix, s) for s in seeds]

    # parallel permutation testing
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        random_corr_coef = list(executor.map(_permute_and_max_corr, tasks))

    ci_interval = scipy.stats.t.interval(
        0.99,
        len(random_corr_coef) - 1,
        loc=np.mean(random_corr_coef),
        scale=scipy.stats.sem(random_corr_coef),
    )
    cutoff = ci_interval[1]  # upper bound of CI is used a the threshold
    print(ci_interval)
    corr_mat[corr_mat <= cutoff] = 0
    corr_mat[corr_mat > cutoff] = 1
    print(f"Superstructure has {np.sum(corr_mat)} edges")
    # Save superstructure as graph and edge list
    G_ss = nx.from_numpy_array(corr_mat)
    return corr_mat

def _convert_local_edge_to_graph(partition, local_cd_edges):
    local_cd_graphs = []
    k = list(partition.keys())
    k.sort()
    for i, edges in zip(k, local_cd_edges):
        node_ids = partition[i]
        if len(node_ids) == 1:
            subgraph = nx.DiGraph()
            subgraph.add_nodes_from(node_ids)
        else:
            # causallearn_graph_to_edges function returns local edge names
            subgraph = nx.from_edgelist(edges, create_using=nx.DiGraph)
            subgraph.add_nodes_from(partition[i])
        local_cd_graphs.append(subgraph)
    return local_cd_graphs

def remove_edges_not_in_ss(
    target_graph: nx.DiGraph, ss_graph: nx.DiGraph
) -> nx.DiGraph:
    """
    Remove edges from target_graph which do not exist in ss_graph

    Notes:
        Final fusion step to merge subgraphs. In infinite data limit this is
        done by screening for conflicting edges during union over subgraphs.

    Args:
        target_graph (nx.DiGraph): the target directed graph
        ss_graph (nx.DiGraph): the superstructure

    Returns:
        nx.DiGraph: target_graph with only edges appearing in ss_graph
    """
    # specify edge orientation to avoid issues with orderings of tuples
    ss_edge_set = set(ss_graph.out_edges())
    target_edge_set = set(target_graph.out_edges())
    # find edges in global_graph that are present in ss_graph
    target_edges_in_superstructure = list(
        target_edge_set.intersection(ss_edge_set),
    )
    # reset all edges in global_graph
    target_graph.remove_edges_from(list(target_graph.edges()))
    # add back edges from global_edges_in_superstructure
    target_graph.add_edges_from(target_edges_in_superstructure)
    return target_graph
def _union_with_overlaps(graphs):
    """
    Helper function that reimplements networkx.union_all, except remove the
    requirement that the node sets be disjoint ie we allow for overlapping
    nodes/edges between graphs

    Args:
        graphs (list[nx.DiGraph]): List of graphs to unite.

    Returns:
        A single, united graph with all the nodes and edges from `graphs`.
    """
    R = None
    seen_nodes = set()
    for i, G in enumerate(graphs):
        G_nodes_set = set(G.nodes)
        if i == 0:
            # Union is the same type as first graph
            R = G.__class__()
        seen_nodes |= G_nodes_set
        R.graph.update(G.graph)
        R.add_nodes_from(G.nodes(data=False))

        if G.is_multigraph():
            edges_to_add = G.edges(keys=True, data=True)
        else:
            edges_to_add = G.edges(data=True)

        R.add_edges_from(edges_to_add)

    if R is None:
        raise ValueError("cannot apply union_all to an empty list")

    return R

def screen_projections_pag2cpdag(
    ss: np.ndarray,
    partition: dict[Any, Any],
    local_cd_graphs: List[GeneralGraph],
    ss_subset: bool = True,
    finite_lim: bool = True,
    data: np.ndarray = None,
    full_cand_set: bool = False,
) -> List[tuple[Any,Any]]:
    
    # The pag represetation has following edge to number mapping 
    # pag[i,j] = 0 iff no edge btw i,j
    # pag[i,j] = 1 iff i *-o j
    # pag[i,j] = 2 iff i *-> j
    # pag[i,j] = 3 iff i *-- j

    # CPDAG
    # cpdag[i,j] = 0 and cpdag[j,i] = 0 iff no edge between i, j
    # cpdag[i,j] = 1 and cpdag[j,i] = 0 iff i->j
    # cpdag[i,j] = 1 and cpdag[j,i] = 1 iff i--j

    # Start with an empty global CPDAG
    cpdag = np.zeros(ss.shape)
    pag_edges = dict()
    for comm_id, pag in enumerate(local_cd_graphs):
        for source, target in itertools.product(pag.nodes, pag.nodes
       ):
            # global_row = partition[comm_id][row]
            # global_col = partition[comm_id][col]
            # # If edge exists in local pag, add an undirected edge in cdpag
            # if pag[row,col] > 0:
            #     cpdag[global_row, global_col] = 1

            # Add edges to dictionary, create a list for overlapping edges
            if (source, target) in pag_edges:
                pag_edges[(source, target)] += [pag[source, target]]
            else:
                pag_edges[(source, target)] = [pag[source, target]]

    # Add all adjacencies that agree
    for edge, end_marks in pag_edges.items():
        u = edge[0]
        v = edge[1]
        # # Tag PAG arrowheads in global CDPAG
        # if any(x==2 for x in end_marks):
        #     cpdag[u,v]=2
        # Add undirected edges
        if all((x !=0) for x in end_marks):
            cpdag[u, v] = 1
            cpdag[v, u] = 1
        # Otherwise, there is disagreement in edge type so remove the edge
        else:
            cpdag[u, v] = 0

    # Find all unshielded colliders in local estimated graphs
    for comm_id, pag in enumerate(local_cd_adj_mats): # TODO fix this
        arrowheads_from = [
            [i for i in range(pag.shape[0]) if pag[i, col] == 2]
            for col in range(pag.shape[1])
        ]
        for col in range(pag.shape[1]):
            # Check if there is a triple
            if len(arrowheads_from[col]) == 2:
                u, v = arrowheads_from[col]
                # check if unshielded and agrees across subsets
                if pag[u, v] == 0:
                    global_u = partition[comm_id][u]
                    global_v = partition[comm_id][v]
                    global_col = partition[comm_id][col]
                    if (
                        cpdag[global_u, global_col] == 1
                        and cpdag[global_col, global_v] == 1
                    ):
                        cpdag[global_u, global_col] = 1
                        cpdag[global_v, global_col] = 1
                        cpdag[global_col, global_u] = 0
                        cpdag[global_col, global_v] = 0
    cpdag_digraph = nx.from_numpy_array(cpdag, create_using=nx.DiGraph)
    # Remove all edges not present in superstructure
    if ss_subset:
        ss_graph = nx.from_numpy_array(ss, create_using=nx.DiGraph)
        cpdag_digraph = remove_edges_not_in_ss(cpdag_digraph, ss_graph)

    if finite_lim:
        cpdag_digraph = screen_projections_finite_lim_postprocessing(
            ss_graph,
            cpdag_digraph,
            partition,
            ss_subset,
            data,
        )
    return list(cpdag_digraph.edges())


def screen_projections(
    ss: np.ndarray,
    partition: dict[Any, Any],
    local_cd_edges: List[Any],
    ss_subset: bool = True,
    finite_lim: bool = True,
    data: np.ndarray = None,
    full_cand_set: bool = False,
) -> List[tuple[Any,Any]]:
    """
    Merge DAG subgraphs by taking the union and resolving conflicts by favoring no
    edge over directed edge. 

    Args:
        ss (np.ndarray): adjacency matrix for the super structure
        partition (dict[Any, Any]): the partition as a dictionary
            {comm_id : [nodes]}
        local_cd_adj_mats (list[np.ndarray]): list of adjacency matrices for
            each local subgraph
        ss_subset (bool): whether to only include edges in global_graph which
            are in ss.
        finite_lim (bool): whether to include adaptations to finite limit
            setting, including resolving bidirected edges using RIC score and
            cycle detection/deletion.
        data (None or np.ndarray): if finite_lim==True, we need data to use
            RIC score.
        full_cand_set (bool): ignore, unused flag.
    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Take the union over graphs
    local_cd_graphs = _convert_local_edge_to_graph(
        partition,
        local_cd_edges,
    )
    global_graph = _union_with_overlaps(local_cd_graphs)
    print(f"Global graph union num edge {len(global_graph.edges())}")

    # Remove all edges not present in superstructure
    ss_graph = nx.from_numpy_array(ss, create_using=nx.DiGraph)
    if ss_subset:
        global_graph = remove_edges_not_in_ss(global_graph, ss_graph)

    # global_graph = no edge if (no edge in comm1) or (no edge in comm2)
    k = list(partition.keys())
    k.sort() 
    for i, edges in zip(k, local_cd_edges):
        comm = partition[i]
        if len(comm) > 1:
            for row, col in itertools.product(
                np.arange(len(comm)), np.arange(len(comm))
            ):
                i = comm[row]
                j = comm[col]
                # if (
                #     not adj_comm[row, col] and not adj_comm[col, row]
                # ) and global_graph.has_edge(i, j):
                #     global_graph.remove_edge(i, j)
                if (
                    (i, j) not in edges and (j,i) not in edges
                ) and global_graph.has_edge(i, j):
                    global_graph.remove_edge(i, j)

    # resolve bidirected edges and delete cycles using RIC score
    if finite_lim:
        global_graph = screen_projections_finite_lim_postprocessing(
            ss_graph,
            global_graph,
            partition,
            ss_subset,
            data,
        )

    return list(global_graph.edges())
    
def screen_projections_finite_lim_postprocessing(
    ss_graph: nx.DiGraph,
    global_graph: nx.DiGraph,
    partition: dict[Any, Any],
    ss_subset=True,
    data=None,
) -> nx.DiGraph:
    """
    Adapts results of screen_projections to finite limit setting by resolving
    bidirected edges using RIC score and cycle detection/deletion.

    Args:
        ss_graph (nx.DiGraph): directed graph for super structure
        global_graph (nx.DiGraph): estimated directed graph
        partition (dict[Any, Any]): the partition as a dictionary
            {comm_id : [nodes]}
        ss_subset (bool): whether to only include edges in global_graph
            which are in ss.
        data (None or np.ndarray): we need data to use RIC score

    Returns:
        nx.DiGraph: the final estimated global directed graph
    """
    # We'll need correlation from data to compute RIC score
    cor = np.corrcoef(data.T)

    # Remove trivial cycles (self-loops)
    global_graph.remove_edges_from(nx.selfloop_edges(global_graph))

    # If no cycles remain, return graph
    # nx.find_cycle will throw an error nx.exception.NetworkXNoCycle
    # if no cycles contained.
    try:
        cycle_list = nx.find_cycle(global_graph, orientation="original")
    except BaseException:
        return global_graph

    # Otherwise, find and eliminate cycles by deleting edges between nodes
    # in overlap. Start by making a dictionary
    # (node_id: list_of_communities_containing_node).
    nodes = list(global_graph.nodes())
    print(nodes) # for some reason there are nodes in the partition that are not in the global graph
    node_to_partition = dict(zip(nodes, [[] for _ in np.arange(len(nodes))]))
    for key, value in partition.items():
        for node in value:
            node_to_partition[node] += [key]

    # Find nodes in the overlap based on this dictionary
    def _find_overlaps(partition):
        overlaps = []
        for node, comm in partition.items():
            if len(comm) > 1:
                overlaps.append(node)

        return overlaps

    overlaps = _find_overlaps(node_to_partition)
    # While the graph contains cycles,
    while len(cycle_list) > 0:
        # If we've found a trivial cycle, i.e (i,j) and (j,i) both exist
        if len(cycle_list) == 2:
            # find the endpoints
            i = cycle_list[0][0]
            j = cycle_list[0][1]

            # remove the edges in question so that predecessors
            # don't include i and j
            global_graph.remove_edge(i, j)
            global_graph.remove_edge(j, i)

            # find parents and use RIC score method
            pa_i = list(global_graph.predecessors(i))
            pa_j = list(global_graph.predecessors(j))
            edge = _resolve_w_ric_score(
                global_graph,
                data,
                cor,
                i,
                j,
                pa_i,
                pa_j,
            )

            # subset_check is true if either we're NOT restricting our
            # estimate to edges present in the superstructure, or if we are
            # restricting and the candidate edge does appear in the
            # superstructure
            subset_check = (not ss_subset) or (edge in list(ss_graph.edges()))
            if edge and subset_check:
                global_graph.add_edge(edge[0], edge[1])
        # If we've found a longer, nontrivial cycle
        else:
            # Find edges that exist in overlap
            edges_in_overlap = []
            for edge in cycle_list:
                # if both endpoints live in overlap
                if edge[0] in overlaps and edge[1] in overlaps:
                    edges_in_overlap.append(edge[:2])
            # Haven't implemented "select edge from overlap with worst METRIC"
            if False:
                # Compute log-likelihood for each edge, and discard the edge
                # with the largest log-likelihood score, i.e. the lowest
                # likelihood score.
                loglikelihood_scores = []
                for edge in edges_in_overlap:
                    i = edge[0]
                    j = edge[1]
                    pa_i = list(global_graph.predecessors(i))
                    pa_j = list(global_graph.predecessors(j))
                    loglikelihood_scores.append(
                        _loglikelihood(data, j, pa_j + [i], cor)
                    )
            # Currently: select arbitrary edge in cycle that's in overlap
            if True:
                if len(edges_in_overlap) == 0:
                    print(
                        "WARNING: CYCLE OCCURS NOT IN OVERLAP. "
                        "Removing arbitrary edge."
                    )
                    edge_data = cycle_list[0]
                else:
                    edge_data = edges_in_overlap[0]
            global_graph.remove_edge(edge_data[0], edge_data[1])

        # nx.find_cycle will throw an error nx.exception.NetworkXNoCycle
        # when all cycles have been removed
        try:
            cycle_list = nx.find_cycle(global_graph, orientation="original")
        except BaseException:
            break

    return global_graph


# For a candidate added edge (u,v)
# if the path v->u exists is the graph, then adding the edge will create
# a cycle
def _detect_cycle(G, edge):
    has_path = nx.has_path(G, edge[1], edge[0])
    return has_path



def _loglikelihood(samples, node, parents, correlation):
    """
    Calculate the the log likelihood of the least squares estimate of a node
    given it's parents

    Args:
        samples (np.ndarray): data matrix where each column corresponds to a
            random variable.
        node (int):the variable (column in data matrix) to calculate the
            likelhood of.
        parents (list of ints): the list of parent ids for the node
        correlation (np.ndarray): the correlation coefficient matrix for the
            data matrix.

    Returns:
        (float) log likelhood value
    """
    cor_nn = correlation[np.ix_([node], [node])]
    cor_pn = correlation[np.ix_(parents, [node])]
    cor_pp = correlation[np.ix_(parents, parents)]
    rss = cor_nn - cor_pn.T.dot(np.linalg.inv(cor_pp)).dot(cor_pn)
    N = samples.shape[0]
    return 0.5 * (
        -N * (np.log(2 * np.pi) + 1 - np.log(N) + np.log(rss * (N - 1)))
    )
    
# From paper https://www.jmlr.org/papers/v21/19-318.html
# Choose no edge, i->j, or j->i based on local RIC score
# Only add an edge if the RIC score for i->j and j->i both are greater
# than the score with no edge
# max(RIC(i->j), RIC(j->i)) < RIC(i,j))
def _resolve_w_ric_score(G, data, cov, i, j, pa_i, pa_j):
    """
    ...

    Args:
        G ():
        data ():
        cov ():
        i ():
        j ():
        pa_i ():
        pa_j ():

    Returns:
        ...
    """
    l_0i = _loglikelihood(data, i, pa_i, cov)
    l_0j = _loglikelihood(data, j, pa_j, cov)
    l_ij = _loglikelihood(data, j, pa_j + [i], cov)
    l_ji = _loglikelihood(data, i, pa_i + [j], cov)
    p = data.shape[1]
    n = data.shape[0]
    lam = np.log(n) if p <= np.sqrt(n) else 2 * np.log(p)

    add_edge = 2 * np.min([l_ij - l_0j, l_ji - l_0i]) > lam

    # Choose the edge that does not result in a cycle, otherwise choose the
    # minimal scoring edge
    if add_edge:
        if _detect_cycle(G, (i, j)):
            return (j, i)
        elif _detect_cycle(G, (j, i)):
            return (i, j)
        elif l_ji > l_ij:
            return (i, j)
        else:
            return (j, i)
    else:
        return None
    
def remove_lowly_expressed_genes(expression_matrix: np.array, gene_names: List[str], expression_threshold=0.8):
    """Remove genes with low expression.

    Args:
        expression_matrix (np.array): Expression matrix with cells as index and 
        genes_names: name of genes corresponding to the columns of the expression matrix
        expression_threshold (float, optional): Min level of expression across cells in percent. Defaults to 0.8.

    Returns:
        DataFrame: Expression matrix with only highly expressed genes
    """
    min_cells = expression_matrix.shape[0] * expression_threshold
    expression_matrix = pd.DataFrame(expression_matrix, columns=gene_names)
    subsample_genes = scprep.filter.filter_rare_genes(
        expression_matrix, min_cells=int(min_cells))
    return subsample_genes.to_numpy(), subsample_genes.columns.to_list()

def adj_to_edge(
    adj: np.ndarray, nodes: list[str], ignore_weights: bool = False
):
    r"""
    Helper function to convert an adjacency matrix into an edge list.
    Optionally include weights so that the edge tuple is (i, j, weight).

    Args:
        adj (np.ndarray): Adjacency  matrix of dimensionality $p \times p$.
        nodes (list[str]): List of node names—in order corresponding to
            rows/cols of `adj`.
        ignore_weights (bool): Ignore the weights if `True`; include them if
            `False`. Defaults to `False`.

    Returns:
        Edge list (of nonzero values) from the given adjacency matrix.
    """
    edges = []
    for row, col in itertools.product(
        np.arange(adj.shape[0]), np.arange(adj.shape[1])
    ):
        if adj[row, col] != 0:
            if ignore_weights:
                edges.append((nodes[row], nodes[col]))
            else:
                edges.append(
                    (nodes[row], nodes[col], {"weight": adj[row, col]})
                )
    return edges

