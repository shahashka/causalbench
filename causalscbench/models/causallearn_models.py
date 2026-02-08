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

import causallearn.search.ConstraintBased.PC
import causallearn.search.ConstraintBased.FCI
import causallearn.search.ScoreBased.GES
import numpy as np
import random 
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    causallearn_graph_to_edges, partion_network, 
    correlation_superstructure, 
    expansive_causal_partition, screen_projections, 
    remove_lowly_expressed_genes)
class GES(AbstractInferenceModel):
    def __init__(self, partition: str = 'disjoint') -> None:
        super().__init__()
        self.partition = partition

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0
    ) -> List[Tuple]:
        if not training_regime == TrainingRegime.Observational:
            return []
        expression_matrix, gene_names = remove_lowly_expressed_genes(

            expression_matrix, gene_names, expression_threshold=0.5
        )
        gene_names = np.array(gene_names)

        def process_partition(partition):
            if len(partition) == 1:
                return []
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:, partition]
            res_map = causallearn.search.ScoreBased.GES.ges(
                expression_matrix_,
                score_func="local_score_BIC",
                maxP=10,
                parameters=None,
            )
            G = res_map["G"]
            return causallearn_graph_to_edges(G, gene_names_)

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
            #partitions = expansive_causal_partition(ss, gene_names=gene_names,resolution=5,cutoff=10, best_n=10)
            partitions = expansive_causal_partition(ss,gene_names=gene_names, resolution=10,cutoff=30, best_n=30)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            edges = []
            with ThreadPoolExecutor(max_workers=2*multiprocessing.cpu_count()) as executor:
                partition_results = list(executor.map(process_partition, partition_inds))
                for result in partition_results:
                    edges.append(result)
            return screen_projections(ss, partitions, edges, ss_subset=True, finite_lim=False)
        else:
            print("GES algorithm must have disjoint or causal partitions")
            NotImplementedError()


class PC(AbstractInferenceModel):
    def __init__(self, partition: str = 'disjoint', missing_value: bool = False) -> None:
        super().__init__()
        self.partition = partition
        self.missing_value = missing_value

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

            expression_matrix, gene_names, expression_threshold=0.5
        )
        gene_names = np.array(gene_names)

        def process_partition(partition):
            if len(partition) == 1:
                return []
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:, partition]
            res = causallearn.search.ConstraintBased.PC.pc(
                expression_matrix_, node_names=gene_names_, mvpc=self.missing_value
            )
            return causallearn_graph_to_edges(res.G, None)
     
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
            # FOR DEBUGGING ss = correlation_superstructure(expression_matrix, seed=seed, num_iterations=10)
            partitions = expansive_causal_partition(ss,gene_names=gene_names, resolution=10,cutoff=30, best_n=30)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            edges = []
            with ThreadPoolExecutor(max_workers=2*multiprocessing.cpu_count()) as executor:
                partition_results = list(executor.map(process_partition, partition_inds))
                for result in partition_results:
                    edges.append(result)
            final_edges = screen_projections(ss, partitions, edges, data=expression_matrix, ss_subset=False, finite_lim=True)
            return final_edges
        else:
            print("PC algorithm must have disjoint or causal partitions")
            NotImplementedError()

    
class FCI(AbstractInferenceModel):
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
        if not training_regime == TrainingRegime.Observational:
            return []
        expression_matrix, gene_names = remove_lowly_expressed_genes(

            expression_matrix, gene_names, expression_threshold=0.5
        )
        gene_names = np.array(gene_names)

        def process_partition(partition):
            if len(partition) == 1:
                return []
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:, partition]
            res = causallearn.search.ConstraintBased.FCI.fci(
                expression_matrix_, node_names=gene_names_, mvpc=self.missing_value
            )
            return causallearn_graph_to_edges(res.G, None)

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
            #partitions = expansive_causal_partition(ss, gene_names=gene_names,resolution=5,cutoff=10, best_n=10)
            partitions = expansive_causal_partition(ss,gene_names=gene_names, resolution=10,cutoff=30, best_n=30)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            edges = []
            with ThreadPoolExecutor(max_workers=2*multiprocessing.cpu_count()) as executor:
                partition_results = list(executor.map(process_partition, partition_inds))
                for result in partition_results:
                    edges.append(result)
            return screen_projections(ss, partitions, edges, ss_subset=True, finite_lim=False)
        else:
            print("FCI algorithm must have disjoint or causal partitions")
            NotImplementedError()
