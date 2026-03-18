import pandas as pd
import numpy as np
from causalscbench.models.utils.model_utils import screen_projections
import pickle
import networkx as nx
from causalscbench.apps.main_app import MainApp
import itertools

# edges = [[], [(np.str_('ENSG00000168028'), np.str_('ENSG00000174748')), (np.str_('ENSG00000168028'), np.str_('ENSG00000144713')), (np.str_('ENSG00000063177'), np.str_('ENSG00000083845')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748'))], [], [(np.str_('ENSG00000167526'), np.str_('ENSG00000164587')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000188846'), np.str_('ENSG00000144713')), (np.str_('ENSG00000168028'), np.str_('ENSG00000188846')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000149273'), np.str_('ENSG00000137154')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [], [(np.str_('ENSG00000167526'), np.str_('ENSG00000147604'))], [(np.str_('ENSG00000138778'), np.str_('ENSG00000088325'))], [(np.str_('ENSG00000168028'), np.str_('ENSG00000188846')), (np.str_('ENSG00000142676'), np.str_('ENSG00000090273')), (np.str_('ENSG00000164587'), np.str_('ENSG00000113013')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000112306'), np.str_('ENSG00000164587')), (np.str_('ENSG00000174748'), np.str_('ENSG00000188846')), (np.str_('ENSG00000144713'), np.str_('ENSG00000188846')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000164587'), np.str_('ENSG00000113013')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [], [], [], [], [(np.str_('ENSG00000142676'), np.str_('ENSG00000164587')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000108298'), np.str_('ENSG00000105372')), (np.str_('ENSG00000144713'), np.str_('ENSG00000114391')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000188846'), np.str_('ENSG00000174748')), (np.str_('ENSG00000168028'), np.str_('ENSG00000188846'))], [], [(np.str_('ENSG00000188846'), np.str_('ENSG00000168028')), (np.str_('ENSG00000174748'), np.str_('ENSG00000188846')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000149273'), np.str_('ENSG00000254772'))], [], [(np.str_('ENSG00000168028'), np.str_('ENSG00000197958'))], [(np.str_('ENSG00000145425'), np.str_('ENSG00000164163')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [], [(np.str_('ENSG00000142676'), np.str_('ENSG00000117410')), (np.str_('ENSG00000105372'), np.str_('ENSG00000063177')), (np.str_('ENSG00000108298'), np.str_('ENSG00000105372')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748'))], [], [(np.str_('ENSG00000112306'), np.str_('ENSG00000008018')), (np.str_('ENSG00000188846'), np.str_('ENSG00000168028')), (np.str_('ENSG00000108298'), np.str_('ENSG00000170889')), (np.str_('ENSG00000142676'), np.str_('ENSG00000090273')), (np.str_('ENSG00000144713'), np.str_('ENSG00000188846')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000105372'), np.str_('ENSG00000063177')), (np.str_('ENSG00000108298'), np.str_('ENSG00000122026')), (np.str_('ENSG00000108298'), np.str_('ENSG00000105372')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000231500'), np.str_('ENSG00000145592'))], [], [(np.str_('ENSG00000112306'), np.str_('ENSG00000008018')), (np.str_('ENSG00000174748'), np.str_('ENSG00000188846')), (np.str_('ENSG00000145425'), np.str_('ENSG00000109475')), (np.str_('ENSG00000144713'), np.str_('ENSG00000188846')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [], [], [(np.str_('ENSG00000142676'), np.str_('ENSG00000122406')), (np.str_('ENSG00000122406'), np.str_('ENSG00000265681'))], [], [], [(np.str_('ENSG00000168028'), np.str_('ENSG00000188846')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000108298'), np.str_('ENSG00000122406')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000164587'), np.str_('ENSG00000113013')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230')), (np.str_('ENSG00000163682'), np.str_('ENSG00000167526'))], [(np.str_('ENSG00000112306'), np.str_('ENSG00000008018')), (np.str_('ENSG00000188846'), np.str_('ENSG00000168028')), (np.str_('ENSG00000174748'), np.str_('ENSG00000188846')), (np.str_('ENSG00000142676'), np.str_('ENSG00000117410')), (np.str_('ENSG00000142676'), np.str_('ENSG00000090273')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000231500'), np.str_('ENSG00000163682')), (np.str_('ENSG00000164587'), np.str_('ENSG00000113013')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [], [(np.str_('ENSG00000188846'), np.str_('ENSG00000168028')), (np.str_('ENSG00000174748'), np.str_('ENSG00000188846')), (np.str_('ENSG00000145425'), np.str_('ENSG00000164163')), (np.str_('ENSG00000145425'), np.str_('ENSG00000109475')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000231500'), np.str_('ENSG00000105372')), (np.str_('ENSG00000231500'), np.str_('ENSG00000147604')), (np.str_('ENSG00000231500'), np.str_('ENSG00000163682')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [], [(np.str_('ENSG00000112306'), np.str_('ENSG00000008018')), (np.str_('ENSG00000105372'), np.str_('ENSG00000063177')), (np.str_('ENSG00000108298'), np.str_('ENSG00000105372')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000164587'), np.str_('ENSG00000113013')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000142676'), np.str_('ENSG00000125691')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000142676'), np.str_('ENSG00000090273')), (np.str_('ENSG00000105372'), np.str_('ENSG00000063177')), (np.str_('ENSG00000108298'), np.str_('ENSG00000105372')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000164587'), np.str_('ENSG00000113013')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000112306'), np.str_('ENSG00000008018')), (np.str_('ENSG00000108298'), np.str_('ENSG00000105372')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [], [(np.str_('ENSG00000188846'), np.str_('ENSG00000114391')), (np.str_('ENSG00000168028'), np.str_('ENSG00000188846')), (np.str_('ENSG00000108298'), np.str_('ENSG00000171863')), (np.str_('ENSG00000164587'), np.str_('ENSG00000113013')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000108298'), np.str_('ENSG00000231500')), (np.str_('ENSG00000142676'), np.str_('ENSG00000090273')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000231500'), np.str_('ENSG00000161970')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000112306'), np.str_('ENSG00000008018')), (np.str_('ENSG00000105372'), np.str_('ENSG00000063177')), (np.str_('ENSG00000108298'), np.str_('ENSG00000105372')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748'))], [(np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000105372'), np.str_('ENSG00000063177')), (np.str_('ENSG00000108298'), np.str_('ENSG00000122026')), (np.str_('ENSG00000108298'), np.str_('ENSG00000105372')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748'))], [(np.str_('ENSG00000112306'), np.str_('ENSG00000008018')), (np.str_('ENSG00000188846'), np.str_('ENSG00000168028')), (np.str_('ENSG00000174748'), np.str_('ENSG00000188846')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000174748'), np.str_('ENSG00000188846')), (np.str_('ENSG00000145425'), np.str_('ENSG00000164163')), (np.str_('ENSG00000145425'), np.str_('ENSG00000109475')), (np.str_('ENSG00000142676'), np.str_('ENSG00000090273')), (np.str_('ENSG00000138778'), np.str_('ENSG00000088325')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000231500'), np.str_('ENSG00000163682')), (np.str_('ENSG00000164587'), np.str_('ENSG00000113013')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [(np.str_('ENSG00000112306'), np.str_('ENSG00000008018')), (np.str_('ENSG00000105372'), np.str_('ENSG00000063177')), (np.str_('ENSG00000108298'), np.str_('ENSG00000105372')), (np.str_('ENSG00000144713'), np.str_('ENSG00000174748')), (np.str_('ENSG00000164587'), np.str_('ENSG00000169230'))], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
with open("/lus/grand/projects/FRAME-IDP/shahashka/causalbench/run_output_notears/343124/local_edges.pkl", "rb") as f:
    edges = pickle.load(f)
output_network = list(set([e for edge in edges for e in edge]))  # flatten

with open('/lus/grand/projects/FRAME-IDP/shahashka/causalbench/run_output_notears/343124/partition.pkl', 'rb') as f:
    partition = pickle.load(f)

for key, value in partition.items():
    if len(value) ==1 :
        print(value)
# Create app and load data + evaluators (required before running evaluation pipeline)
main_app = MainApp(
    output_directory="./debugging",
    data_directory="./causalscbench/data_access/data",
    model_name="notears-lin",
    dataset_name="weissmann_k562",
    max_path_length=-1,
    omission_estimation_size=0,
    subset_data=1.0,
    model_seed=0,
    filter=True,
)

main_app.load_data()
main_app.load_evaluators()
(
    expression_matrix_train,
    interventions_train,
    gene_names,
) = main_app.dataset_splitter.get_observational()


def _union_with_overlaps(graphs):
    """
    Helper that reimplements networkx union_all but allows overlapping node sets.
    """
    R = None
    for i, G in enumerate(graphs):
        if i == 0:
            R = G.__class__()
        R.graph.update(G.graph)
        R.add_nodes_from(G.nodes(data=False))
        edges_to_add = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
        R.add_edges_from(edges_to_add)
    if R is None:
        raise ValueError("cannot apply union_all to an empty list")
    return R


def screen_edge_to_graph(partition, local_cd_edges):
    local_cd_graphs = []
    k = list(partition.keys())
    k.sort()
    for i, edge_list in zip(k, local_cd_edges):
        node_ids = partition[i]
        if len(node_ids) == 1:
            subgraph = nx.DiGraph()
            subgraph.add_nodes_from(node_ids)
        else:
            subgraph = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
            subgraph.add_nodes_from(partition[i])
        local_cd_graphs.append(subgraph)
    return local_cd_graphs


def screen(partition, global_graph, local_cd_edges):
    screened_edges = []
    k = list(partition.keys())
    k.sort()
    for p, edge_list in zip(k, local_cd_edges):
        comm = partition[p]
        assert all((e[0] in comm) and (e[1] in comm) for e in edge_list)
        if len(comm) > 1:
            for row, col in itertools.product(
                np.arange(len(comm)), np.arange(len(comm))
            ):
                i, j = comm[row], comm[col]
                if (i, j) not in edge_list and (j, i) not in edge_list and global_graph.has_edge(i, j):
                    screened_edges.append((i, j))
                    global_graph.remove_edge(i, j)
    return list(global_graph.edges()), screened_edges


out_graphs = screen_edge_to_graph(partition, edges)
global_graph = _union_with_overlaps(out_graphs)
print(len(global_graph.edges()), len(output_network))
out_edges, screened_edges = screen(partition, global_graph, edges)
print(out_edges, len(out_edges))
for edge in screened_edges:
    num_graphs = 0
    num_graphs_with_edge = 0
    for g in out_graphs:
        if edge[0] in g.nodes() and edge[1] in g.nodes():
            num_graphs += 1
            if edge in g.edges():
                num_graphs_with_edge += 1
    ratio = num_graphs_with_edge / num_graphs if num_graphs else 0
    print(edge[0], edge[1], num_graphs, num_graphs_with_edge, ratio)
    if ratio >= 0.5:
        out_edges.append(edge)

# Run the same evaluation pipeline as main_app.train_and_evaluate()
corum_evaluation = main_app.corum_evaluator.evaluate_network(out_edges)
ligand_receptor_evaluation = main_app.lr_evaluator.evaluate_network(out_edges)
string_network_evaluation = main_app.string_network_evaluator.evaluate_network(out_edges)
string_physical_evaluation = main_app.string_physical_evaluator.evaluate_network(out_edges)
chipseq_evaluation = main_app.chipseq_evaluator.evaluate_network(out_edges, directed=True)
pooled_biological_evaluation = main_app.pooled_biological_evaluator.evaluate_network(out_edges)
pooled_biological_sigificant_evaluation = main_app.pooled_biological_significant_evaluator.evaluate_network(
    out_edges, directed=True
)
quantitative_test_evaluation = main_app.quantitative_evaluator.evaluate_network(
    out_edges,
    main_app.max_path_length,
    main_app.check_false_omission_rate,
    main_app.omission_estimation_size,
    seed=main_app.model_seed,
)
metrics = {
    "corum_evaluation": corum_evaluation,
    "ligand_receptor_evaluation": ligand_receptor_evaluation,
    "quantitative_test_evaluation": quantitative_test_evaluation,
    "string_network_evaluation": string_network_evaluation,
    "string_physical_evaluation": string_physical_evaluation,
    "chipseq_evaluation": chipseq_evaluation,
    "pooled_biological_evaluation": pooled_biological_evaluation,
    "pooled_biological_sigificant_evaluation": pooled_biological_sigificant_evaluation,
}
print(metrics)
