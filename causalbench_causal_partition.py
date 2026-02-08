from parsl.app.app import python_app
from parsl_setup import get_parsl_config, get_parsl_config_debug
import parsl
import sys
import os
@python_app
def run_experiment(worker_id, cd, dataset, training_regime): 
    import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
    # sys.path.insert(1, '/grand/projects/FRAME-IDP/shahashka/causalbench/')
    import copy
    import warnings
    import logging
    import subprocess
    warnings.simplefilter("ignore") 
    logger = logging.getLogger(__name__)
    outdir='/grand/projects/FRAME-IDP/shahashka/causalbench/run_output_parsl'
    datadir = "/grand/projects/FRAME-IDP/shahashka/causalbench/causalscbench/data_access/data"
    logging.basicConfig(
        filename=f"{outdir}/worker_{worker_id}.stderr",  # Specify the log file name
        level=logging.DEBUG,  # Set the logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s' # Define the log message format
    )

    logger.debug(f"Run experiment {cd}, {training_regime}, {dataset}")
    #causalbench_run   --dataset_name weissmann_k562   --output_directory ./run_output_parsl     --data_directory ./causalscbench/data_access/data     --training_regime observational     --model_name notears-lin_disjoint_part     --subset_data 1.0     --model_seed 0     --do_filter     --max_path_length -1     --omission_estimation_size 500
    try:
        subprocess.run(["causalbench_run", "--dataset_name",dataset, "--output_directory", outdir, "--data_directory", datadir, "--training_regime", training_regime, "--model_name", cd,  "--model_seed", "0","--do_filter", "--max_path_length", "-1", '--omission_estimation_size', "500" ]) 
    except Exception as e:
        print(f"Command failed with error: {e}")
        
    return worker_id, cd, dataset, training_regime
if __name__ == '__main__':
    #config = get_parsl_config_debug()
    config = get_parsl_config()
    # NOTE(MS): this is how you vary arg inputs into your expeirment
    args = []
    dataset_name=['weissmann_k562']#, "weissmann_rpe1"]
    cd_name_obs = [
            "random100",
            "random1000",
            "random10000",
            "fully-connected",
            "lasso",
            "random_forest",
            "grnboost",
            "genie",
            "ges", # partitioned size 30 
            "ges_causal_part",
            "pc",# partitioned size 30 
            "pc_causal_part",
            "gsp",
            "gsp_disjoint_part",
            "gsp_causal_part",
            "notears-lin",
            "notears-lin_disjoint_part",
            "notears-lin_causal_part",
            "notears-lin-sparse",
            "notears-lin-sparse_disjoint_part",
            "notears-lin-sparse_causal_part",
            "notears-mlp",
            "notears-mlp_disjoint_part",
            "notears-mlp_causal_part",
            "notears-mlp-sparse",
            "notears-mlp-sparse_disjoint_part",
            "notears-mlp-sparse_causal_part",
            
            ]
    cd_name_int = [
            "random100",
            "random1000",
            "random10000",
            "gies", # partitioned size 30 
            "gies_causal_part",
            "igsp",
            "igsp_disjoint_part",
            "igsp_causal_part",
            "DCDI-G",# partitioned size 50 
            "DCDI-G_causal_part",
            "DCDI-DSF", # partitioned size 50 
            "DCDI-DSF_causal_part",
            # "DCDFG-LIN", # partitioned size 50 
            # "DCDFG-MLP" # partitioned size 50 
] # 41 cd algs * 2 datasets 
    n_repeats = 1 #5
    worker_id = 0
    for n in range(n_repeats):
        for d in dataset_name:
            for cd in cd_name_int:
                args.append({"worker_id":worker_id, "cd":cd, "dataset":d, 'training_regime':"interventional"})
                worker_id += 1
                print(worker_id)
                
            for cd in cd_name_obs:
                args.append({"worker_id":worker_id, "cd":cd, "dataset":d, 'training_regime':"observational"})
                worker_id += 1
                print(worker_id)
            
    with parsl.load(config):
        # Launch experiments as parsl tasks
        futures = [run_experiment(**arg) for arg in args]

        # Wait for tasks to return
        for future in futures:
            print(f'Waiting for {future}', file=sys.stderr)
            print(f'Got result {future.result()}',  file=sys.stderr)







