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
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    partion_network, 
    correlation_superstructure, 
    expansive_causal_partition, screen_projections, 
    remove_lowly_expressed_genes)

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

        def process_partition(partition):
            if len(partition) == 1:
                return []
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:,partition]
            adjacency = causalscbench.third_party.notears.linear.notears_linear(
                expression_matrix_, lambda1=self.lambda1, max_iter=20, loss_type="l2"
            )
            indices = np.transpose(np.nonzero(adjacency))
            edges_partition = set()
            for (i, j) in indices:
                edges_partition.add((gene_names_[i], gene_names_[j]))
            return list(edges_partition)
        
        if self.partition == 'disjoint':
            partitions = partion_network(gene_names, 30, seed)
            edges = []
            with ThreadPoolExecutor(max_workers=2*multiprocessing.cpu_count()) as executor:
                partition_results = list(executor.map(process_partition, partitions))
                for result in partition_results:
                    edges += result
            return edges
        elif self.partition == 'causal':
            ss = correlation_superstructure(expression_matrix, seed=seed, num_iterations=100)
            partitions = expansive_causal_partition(ss,gene_names=gene_names, resolution=5,cutoff=30, best_n=30)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            edges = []
            with ThreadPoolExecutor(max_workers=2*multiprocessing.cpu_count()) as executor:
                partition_results = list(executor.map(process_partition, partition_inds))
                for result in partition_results:
                    edges.append(result)
            return screen_projections(ss, partitions, edges, ss_subset=True, finite_lim=False)
        else:
            return process_partition([i for i in range(len(gene_names))])


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

        def process_partition(partition):
            if len(partition) == 1:
                return []
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:,partition]
            model = causalscbench.third_party.notears.nonlinear.NotearsMLP(dims=[len(gene_names_), 10, 1], bias=True)
            adjacency = causalscbench.third_party.notears.nonlinear.notears_nonlinear(
                model, expression_matrix_, lambda1=self.lambda1, lambda2=self.lambda1, max_iter=20
            )
            indices = np.transpose(np.nonzero(adjacency))
            edges_partition = set()
            for (i, j) in indices:
                edges_partition.add((gene_names_[i], gene_names_[j]))
            return list(edges_partition)
        
        if self.partition == 'disjoint':
            partitions = partion_network(gene_names, 30, seed)
            edges = []
            with ThreadPoolExecutor(max_workers=2*multiprocessing.cpu_count()) as executor:
                partition_results = list(executor.map(process_partition, partitions))
                for result in partition_results:
                    edges += result
            return edges
        elif self.partition == 'causal':
            ss = correlation_superstructure(expression_matrix, seed=seed, num_iterations=100)
            partitions = expansive_causal_partition(ss,gene_names=gene_names, resolution=5,cutoff=30, best_n=30)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            edges = []
            with ThreadPoolExecutor(max_workers=2*multiprocessing.cpu_count()) as executor:
                partition_results = list(executor.map(process_partition, partition_inds))
                for result in partition_results:
                    edges.append(result)
            return screen_projections(ss, partitions, edges, ss_subset=True, finite_lim=False)
        else:
            return process_partition([i for i in range(len(gene_names))])

