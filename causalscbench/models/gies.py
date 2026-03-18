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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import gies
import numpy as np
import pandas as pd
import multiprocessing
from causalscbench.model.utils import partition_config
from concurrent.futures import ProcessPoolExecutor
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    partion_network, 
    correlation_superstructure, 
    expansive_causal_partition, screen_projections, rand_edge_cover_partition,
    remove_lowly_expressed_genes)

### PARTITION HELPER FUNCTIONS START ###
def process_partition_gies(args):
    partition, gene_names_, expression_matrix_, ss_, interventions = args
    if len(partition) == 1:
        return []
    node_dict = {g: idx for idx, g in enumerate(gene_names_)}
    gene_names_set = set(gene_names_)
    subset = []
    interventions_ = []
    for idx, iv in enumerate(interventions):
        if iv in gene_names_set or iv == "non-targeting":
            subset.append(idx)
            interventions_.append(iv)
    expression_matrix_ = expression_matrix_[subset, :]
    gene_to_interventions = dict()
    for i, intervention in enumerate(interventions_):
        gene_to_interventions.setdefault(intervention, []).append(i)
    data = []
    I = []
    for inv, indices in gene_to_interventions.items():
        if inv == "non-targeting":
            I.append([])
        else:
            I.append([node_dict[inv]])
        data.append(expression_matrix_[indices, :])

    adjacency, _ = gies.fit_bic(data, I, A0=ss_, iterate=False)
    indices = np.transpose(np.nonzero(adjacency))
    edges_partition = set()
    for (i, j) in indices:
        edges_partition.add((gene_names_[i], gene_names_[j]))
    return list(edges_partition)
### PARTITION HELPER FUNCTIONS END ###

class GIES(AbstractInferenceModel):
    def __init__(self, partition: str = 'disjoint') -> None:
        super().__init__()
        self.partition = partition
        
    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        gies.np.bool = bool
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.5
        )
        gene_names = np.array(gene_names)
        interventions = list(interventions)
  
        if self.partition == 'disjoint':
            partitions = partion_network(gene_names, partition_config.SIZE_DISJOINT, seed)
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], None, interventions)
                for partition in partitions
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_gies, tasks):
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
                (partition, gene_names[partition], expression_matrix[:,partition], ss[partition][:,partition], interventions)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_gies, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges, 
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
                (partition, gene_names[partition], expression_matrix[:,partition], ss[partition][:,partition], interventions)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_gies, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges, 
                                         ss_subset=partition_config.SS_SUBSET, 
                                         finite_lim=partition_config.FINITE_LIMIT)
            return {"network": network, "superstructure": ss, "partition": partitions, "local_edges": edges}
        else:
            print("GES algorithm must have disjoint, edge cover or causal partitions")
            NotImplementedError()