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
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import causallearn.search.ConstraintBased.PC
import causallearn.search.ConstraintBased.FCI
import causallearn.search.ScoreBased.GES
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causalscbench.models.utils import partition_config 
import numpy as np
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    causallearn_graph_to_edges, partion_network, 
    correlation_superstructure, 
    expansive_causal_partition, screen_projections, rand_edge_cover_partition,
    remove_lowly_expressed_genes)

### PARTITION HELPER FUNCTIONS START ###
def process_partition_ges(args):
    partition, gene_names_, expression_matrix_, ss_ = args

    if len(partition) == 1:
        return []
    res_map = causallearn.search.ScoreBased.GES.ges(
        expression_matrix_,
        score_func="local_score_BIC",
        maxP=10,
        parameters=None,
        skeleton=ss_
    )
    G = res_map["G"]
    return causallearn_graph_to_edges(G, gene_names_)
    
def process_partition_pc(args):
    partition, gene_names_, expression_matrix_, ss_, missing_value = args

    if len(partition) == 1:
        return []
    if ss_:
        bg = BackgroundKnowledge()
        for i in range(ss_.shape[0]):
            for j in range(ss_.shape[1]):
                if ss_[i,j] == 0:
                    bg.add_forbidden_by_node(gene_names_[i],gene_names_[j])
    else:
        bg = None
    res = causallearn.search.ConstraintBased.PC.pc(
            expression_matrix_, node_names=gene_names_, mvpc=missing_value, background_knowledge=bg
        )
    return causallearn_graph_to_edges(res.G, None)

def process_partition_fci(args):
    partition, gene_names_, expression_matrix_, ss_, missing_value = args

    if len(partition) == 1:
        return []

    if ss_:
        bg = BackgroundKnowledge()
        for i in range(ss_.shape[0]):
            for j in range(ss_.shape[1]):
                if ss_[i,j] == 0:
                    bg.add_forbidden_by_node(gene_names_[i],gene_names_[j])
    else:
        bg = None
    res = causallearn.search.ConstraintBased.FCI.fci(
        expression_matrix_, node_names=gene_names_, mvpc=missing_value, background_knowledge=bg
    )
    return causallearn_graph_to_edges(res.G, None)
### PARTITION HELPER FUNCTIONS END ###

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
        print(f"Number of genes {len(gene_names)}")

        if self.partition == 'disjoint':
            partitions = partion_network(gene_names, partition_config.SIZE_DISJOINT, seed)
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], None)
                for partition in partitions
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_ges, tasks):
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
                (partition, gene_names[partition], expression_matrix[:,partition], ss[partition][:,partition])
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_ges, tasks):
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
                (partition, gene_names[partition], expression_matrix[:,partition], ss[partition][:,partition])
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_ges, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges, 
                                         ss_subset=partition_config.SS_SUBSET, 
                                         finite_lim=partition_config.FINITE_LIMIT)
            return {"network": network, "superstructure": ss, "partition": partitions, "local_edges": edges}
        else:
            print("GES algorithm must have disjoint, edge cover or causal partitions")
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

        if self.partition == 'disjoint':
            partitions = partion_network(gene_names, partition_config.SIZE_DISJOINT, seed)
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], None, self.missing_value)
                for partition in partitions
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_pc, tasks):
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
                (partition, gene_names[partition], expression_matrix[:,partition], ss[partition][:,partition], self.missing_value)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_pc, tasks):
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
                (partition, gene_names[partition], expression_matrix[:,partition], ss[partition][:,partition], self.missing_value)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_pc, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges, 
                                         ss_subset=partition_config.SS_SUBSET,
                                         finite_lim=partition_config.FINITE_LIMIT)
            return {"network": network, "superstructure": ss, "partition": partitions, "local_edges": edges}
        else:
            print("PC algorithm must have disjoint, edge cover or causal partitions")
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

        if self.partition == 'disjoint':
            partitions = partion_network(gene_names, partition_config.SIZE_DISJOINT, seed)
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], None, self.missing_value)
                for partition in partitions
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_fci, tasks):
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
                (partition, gene_names[partition], expression_matrix[:,partition], ss[partition][:,partition], self.missing_value)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_fci, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges,
                                         ss_subset=partition_config.SS_SUBSET,
                                         finite_lim=partition_config.FINITE_LIMIT)
            return {"network": network, "superstructure": ss, "partition": partitions, "local_edges": edges}
        
        elif self.partition == 'edge_cover':
            ss = correlation_superstructure(expression_matrix, seed=seed)#, num_iterations=100)
            partitions = rand_edge_cover_partition(ss,gene_names=gene_names,
                                                    resolution=partition_config.RESOLUTION,
                                                    cutoff=partition_config.CUTOFF, 
                                                    best_n=partition_config.BEST_N)
            partition_inds = [[np.argwhere(gene_names==p)[0][0] for p in part] for part in partitions.values()]
            tasks = [
                (partition, gene_names[partition], expression_matrix[:,partition], ss[partition][:,partition], self.missing_value)
                for partition in partition_inds
            ]
            edges = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for e in executor.map(process_partition_fci, tasks):
                    edges.append(e)
            network = screen_projections(ss, partitions, edges, 
                                         ss_subset=partition_config.SS_SUBSET,
                                         finite_lim=partition_config.FINITE_LIMIT)
            return {"network": network, "superstructure": ss, "partition": partitions, "local_edges": edges}
        else:
            print("FCI algorithm must have disjoint, edge cover or causal partitions")
            NotImplementedError()
