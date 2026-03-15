#!/bin/sh
#PBS -l select=1:system=polaris
#PBS -l walltime=24:00:00
#PBS -l filesystems=grand
#PBS -q preemptable
#PBS -A FRAME-IDP
export OPENBLAS_NUM_THREADS=1
cd /lus/grand/projects/FRAME-IDP/shahashka/
source cb_env/bin/activate
cd causalbench
#echo Hello world!
# causalbench_run   --dataset_name weissmann_k562   --output_directory ./run_output_notears     --data_directory ./causalscbench/data_access/data     --training_regime observational     --model_name notears-lin     --subset_data 1.0     --model_seed 0     --do_filter     --max_path_length -1     --omission_estimation_size 500
causalbench_run   --dataset_name weissmann_k562   --output_directory ./run_output_notears     --data_directory ./causalscbench/data_access/data     --training_regime observational     --model_name notears-lin_causal_part     --subset_data 1.0     --model_seed 0     --do_filter     --max_path_length -1     --omission_estimation_size 500

# causalbench_run   --dataset_name weissmann_k562   --output_directory ./run_output_tuning_notears     --data_directory ./causalscbench/data_access/data     --training_regime observational     --model_name notears-mlp     --subset_data 1.0     --model_seed 0     --do_filter     --max_path_length -1     --omission_estimation_size 500
