"""
Copyright (C) 2022  GlaxoSmithKline plc - Yusuf Roohani;

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

import pdb
import networkx as nx
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from causaldag import (MemoizedCI_Tester, MemoizedInvarianceTester,
                       gauss_invariance_suffstat, gauss_invariance_test, gsp,
                       igsp, partial_correlation_suffstat, rand)

from causalscbench.third_party.causaldag import partial_correlation_test

from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    partion_network, 
    correlation_superstructure, 
    expansive_causal_partition, screen_projections, 
    remove_lowly_expressed_genes)
class GreedySparsestPermutation(AbstractInferenceModel):
    """Network inference model based on GSP."""

    def __init__(self, partition: str = 'none') -> None:
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
            expression_matrix, gene_names, expression_threshold=0.25
        )
        gene_names = np.array(gene_names)

        def process_partition(partition):
            if len(partition) == 1:
                return []
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:,partition]
            nodes = list(range(len(gene_names_)))
            suffstat = partial_correlation_suffstat(expression_matrix_)
            ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3, 
                                        track_times=True)
            dag = gsp(set(nodes), ci_tester, depth=2)
            
            ## Convert edges to correct format
            edges_partition = set()
            for edge in nx.generate_adjlist(dag.to_nx()):
                edge_nodes = [int(e) for e in edge.split(' ')]
                if len(edge_nodes)>1:
                    edges_partition.add((gene_names_[edge_nodes[0]], gene_names_[edge_nodes[1]]))
        
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
            partitions = expansive_causal_partition(ss,gene_names=gene_names, resolution=10,cutoff=30, best_n=30)
            edges = []
            with ThreadPoolExecutor(max_workers=2*multiprocessing.cpu_count()) as executor:
                partition_results = list(executor.map(process_partition, partitions))
                for result in partition_results:
                    edges.append(result)
            return screen_projections(ss, partitions, edges, ss_subset=True, finite_lim=False)
        else:
            return process_partition([i for i in range(len(gene_names))])

class InterventionalGreedySparsestPermutation(AbstractInferenceModel):
    """Network inference model based on GSP."""

    def __init__(self, partition: str = 'none') -> None:
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
        
        edges = set()
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.5
        )
        gene_names = np.array(gene_names)
        interventions = [i for i in interventions]
        # Create list of observational indices
        obs_idxs = np.where(np.array(interventions)=="non-targeting")[0]

        def process_partition(partition):
            if len(partition) == 1:
                return []
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:,partition]
            nodes = list(range(len(gene_names_)))

            # Observational samples
            obs_samples = expression_matrix_[obs_idxs, :]

            # Interventional samples
            interventions_partition = [i for i in interventions if (i in gene_names_) and (i != "non-targeting")]
            interventions_unique = list(set(interventions_partition).difference(set(['non-targeting'])))
            iv_idxs = [np.where(np.array(interventions_partition)==i)[0] for i in interventions_unique]
            
            intv_to_remove = []
            iv_samples_list = []
            node_dict = {g:idx for idx, g in enumerate(gene_names_)}
            # remove lists with only a single sample
            for iv_idx, intv_name in zip(iv_idxs, interventions_unique):
                if len(iv_idx)>1:
                    iv_samples_list.append(expression_matrix_[iv_idx, :])
                else:
                    intv_to_remove.append(intv_name)
            interventions_partition = [x for x in interventions_partition if x not in intv_to_remove]
            interventions_unique = list(set(interventions_unique).difference(set(intv_to_remove)))  
            setting_list = [{'interventions': [node_dict[i]]} for i in interventions_unique]
                        
            # Sufficient statistics for observational and interventional data
            obs_suffstat = partial_correlation_suffstat(obs_samples)
            inv_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)
            
            # CI tester and invariance tester
            ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=1e-3)
            inv_tester = MemoizedInvarianceTester(gauss_invariance_test, inv_suffstat, 
                                                alpha=1e-3)
        
            # Estimate DAG
            dag = igsp(
                setting_list,
                set(nodes),
                ci_tester,
                inv_tester
            )
            
            ## Convert edges to correct format
            edges_partition = set()
            for edge in nx.generate_adjlist(dag.to_nx()):
                edge_nodes = [int(e) for e in edge.split(' ')]
                if len(edge_nodes)>1:
                    edges_partition.add((gene_names_[edge_nodes[0]], gene_names_[edge_nodes[1]]))
        
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
            partitions = expansive_causal_partition(ss,gene_names=gene_names, resolution=10,cutoff=30, best_n=30)
            edges = []
            with ThreadPoolExecutor(max_workers=2*multiprocessing.cpu_count()) as executor:
                partition_results = list(executor.map(process_partition, partitions))
                for result in partition_results:
                    edges.append(result)
            return screen_projections(ss, partitions, edges, ss_subset=True, finite_lim=False)
        else:
            return process_partition([i for i in range(len(gene_names))])
