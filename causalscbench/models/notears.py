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

from typing import List, Tuple

import causalscbench.third_party.notears.linear
import causalscbench.third_party.notears.nonlinear
import causalscbench.third_party.notears.utils
from causalscbench.models.utils import partition_config 
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    partion_network, 
    correlation_superstructure, 
    expansive_causal_partition,
    rand_edge_cover_partition, screen_projections, 
    remove_lowly_expressed_genes)

### PARTITION HELPER FUNCTIONS START ###
def process_partition_lin(args):
    partition, gene_names_, expression_matrix_, lambda1 = args

    if len(partition) == 1:
        return []

    adjacency = causalscbench.third_party.notears.linear.notears_linear(
        expression_matrix_,
        lambda1=lambda1,
        max_iter=20,
        loss_type="l2",
        w_threshold=0.3,
    )
    indices = np.transpose(np.nonzero(adjacency))
    edges_partition = set()
    for (i, j) in indices:
        edges_partition.add((gene_names_[i], gene_names_[j]))
    return list(edges_partition)

def process_partition_mlp(args):
    partition, gene_names_, expression_matrix_, lambda1 = args

    if len(partition) == 1:
        return []
    
    model = causalscbench.third_party.notears.nonlinear.NotearsMLP(dims=[len(gene_names_), 10, 1], bias=True)
    adjacency = causalscbench.third_party.notears.nonlinear.notears_nonlinear(
        model, expression_matrix_, lambda1=lambda1, lambda2=lambda1, max_iter=20, w_threshold=0.3
    )
    indices = np.transpose(np.nonzero(adjacency))
    edges_partition = set()
    for (i, j) in indices:
        edges_partition.add((gene_names_[i], gene_names_[j]))
    return list(edges_partition)
        
### PARTITION HELPER FUNCTIONS END ###

class NotearsLin(AbstractInferenceModel):
    def __init__(self, lambda1: float = 0.0, partition: str = 'none') -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.partition = partition


    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        if not training_regime == TrainingRegime.Observational:
            return []
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.25
        )
        
        causalscbench.third_party.notears.utils.set_random_seed(seed)
        gene_names = np.array(gene_names)
        print(f"Number of genes {len(gene_names)}")
        
        if self.partition == 'disjoint':
            partitions = partion_network(gene_names, partition_config.SIZE_DISJOINT , seed)
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], self.lambda1)
                for partition in partitions
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_lin, tasks):
                    edges += e
            return {"network": edges, "partition": partitions}
        
        elif self.partition == 'causal':
            ss = correlation_superstructure(expression_matrix, seed=seed)
            partitions = expansive_causal_partition(ss,gene_names=gene_names,
                                                    resolution=partition_config.RESOLUTION,
                                                    cutoff=partition_config.CUTOFF, 
                                                    best_n=partition_config.BEST_N)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], self.lambda1)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_lin, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges,data=expression_matrix,
                                         ss_subset=partition_config.SS_SUBSET, 
                                         finite_lim=partition_config.FINITE_LIMIT)
            return {"network": network, "superstructure": ss, "partition": partitions, "local_edges": edges}
        elif self.partition == 'edge_cover':
            ss = correlation_superstructure(expression_matrix, seed=seed)
            partitions = rand_edge_cover_partition(ss,gene_names=gene_names,
                                                    resolution=partition_config.RESOLUTION,
                                                    cutoff=partition_config.CUTOFF, 
                                                    best_n=partition_config.BEST_N)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], self.lambda1)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_lin, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges,data=expression_matrix,
                                         ss_subset=partition_config.SS_SUBSET, 
                                         finite_lim=partition_config.FINITE_LIMIT)
            return {"network": network, "superstructure": ss, "partition": partitions, "local_edges": edges}
       
        else:
            return {"network": process_partition_lin([i for i in range(len(gene_names))])}

class NotearsMLP(AbstractInferenceModel):
    def __init__(self, lambda1: float = 0.0, partition: str = 'none') -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.partition = partition

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        if not training_regime == TrainingRegime.Observational:
            return []
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.25
        )
        causalscbench.third_party.notears.utils.set_random_seed(seed)
        gene_names = np.array(gene_names)

        if self.partition == 'disjoint':
            partitions = partion_network(gene_names, partition_config.SIZE_DISJOINT , seed)
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], self.lambda1)
                for partition in partitions
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_mlp, tasks):
                    edges += e
            return {"network": edges, "partition": partitions}
        
        elif self.partition == 'causal':
            ss = correlation_superstructure(expression_matrix, seed=seed)
            partitions = expansive_causal_partition(ss,gene_names=gene_names,
                                                    resolution=partition_config.RESOLUTION,
                                                    cutoff=partition_config.CUTOFF, 
                                                    best_n=partition_config.BEST_N)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], self.lambda1)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_mlp, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges,data=expression_matrix,
                                         ss_subset=partition_config.SS_SUBSET, 
                                         finite_lim=partition_config.FINITE_LIMIT)
            return {"network": network, "superstructure": ss, "partition": partitions, "local_edges": edges}
        elif self.partition == 'edge_cover':
            ss = correlation_superstructure(expression_matrix, seed=seed)
            partitions = rand_edge_cover_partition(ss,gene_names=gene_names,
                                                    resolution=partition_config.RESOLUTION,
                                                    cutoff=partition_config.CUTOFF, 
                                                    best_n=partition_config.BEST_N)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], self.lambda1)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_mlp, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges,data=expression_matrix,
                                         ss_subset=partition_config.SS_SUBSET, 
                                         finite_lim=partition_config.FINITE_LIMIT)
            return {"network": network, "superstructure": ss, "partition": partitions, "local_edges": edges}
       
        else:
            return {"network": process_partition_mlp([i for i in range(len(gene_names))])}
