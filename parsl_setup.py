"""setting up Parsl utilities for experiment parallelization."""

from __future__ import annotations

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
#from parsl.executors.ipp import IPyParallelExecutor
from parsl.launchers import MpiExecLauncher
from parsl.addresses import address_by_interface, address_by_hostname
from parsl.providers import PBSProProvider
from parsl.monitoring.monitoring import MonitoringHub
from parsl.launchers import SingleNodeLauncher
from parsl.providers import LocalProvider
def get_parsl_config() -> Config:
    """Initialize Parsl config.

    One experiment per GPU.
    Multiple experiment per node.
    """

    # NOTE(MS): replace these
    run_dir = "/grand/projects/FRAME-IDP/shahashka/causalbench/"
    user_opts = {
        "worker_init": f"""
module use /soft/modulefiles
module load spack-pe-base/0.10.1  
source /grand/projects/FRAME-IDP/shahashka/cb_env/bin/activate
cd {run_dir} 
# Print to stdout to for easier debugging
module list
nvidia-smi
which python
hostname
pwd""",
        "scheduler_options": "#PBS -l filesystems=home:grand:grand",  # specify any PBS options here, like filesystems
        "account": " FRAME-IDP",
        "queue": "preemptable",  # e.g.: "prod","debug, "preemptable", "debug-scaling" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "24:00:00", #HH:MM:SS
        "nodes_per_block": 3,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
        "available_accelerators": 4,  # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
    }
    provider=PBSProProvider(
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                    ),
                    account=user_opts["account"],
                    queue=user_opts["queue"],
                    select_options="ngpus=4",
                    # PBS directives (header lines): for array jobs pass '-J' option
                    scheduler_options=user_opts["scheduler_options"],
                    # Command to be run before starting a worker, such as:
                    worker_init=user_opts["worker_init"],
                    # number of compute nodes allocated for each block
                    nodes_per_block=user_opts["nodes_per_block"],
                    init_blocks=1,
                    min_blocks=0,#0,
                    max_blocks=1,  # Can increase more to have more parallel jobs
                    # cpus_per_node=user_opts["cpus_per_node"],
                    walltime=user_opts["walltime"],
                )

    config = Config(
        run_dir="/grand/projects/FRAME-IDP/shahashka/runinfo_causalbench",
        executors=[
            HighThroughputExecutor(
                label='causalbench',
                available_accelerators=4,  # number of GPUs
                max_workers_per_node=4,
                provider=provider,
                #address=address_by_interface("bond0"),
                cpu_affinity="block-reverse",
                prefetch_capacity=0,
            )
        ],
        monitoring=MonitoringHub(
            hub_address=address_by_hostname(),
            monitoring_debug=False,
            resource_monitoring_interval=10,
              ),
    )
    return config

def get_parsl_config_debug() -> Config:
    """Initialize Parsl config.

    One experiment per GPU.
    Multiple experiment per node.
    """

    # NOTE(MS): replace these
    #env = "/grand/projects/FRAME-IDP/shahashka/env"
    run_dir = "/grand/projects/FRAME-IDP/shahashka/causalbench/"
    user_opts = {
        "worker_init": f"""
module use /soft/modulefiles
module load spack-pe-base/0.10.1  
# For internet access
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
source /grand/projects/FRAME-IDP/shahashka/cb_env/bin/activate
cd {run_dir} 
# Print to stdout to for easier debugging
module list
nvidia-smi
which python
hostname
pwd""",
        "scheduler_options": "#PBS -l filesystems=home:grand:grand",  # specify any PBS options here, like filesystems
        "account": " FRAME-IDP",
        "queue": "debug",  # e.g.: "prod","debug, "preemptable", "debug-scaling" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "00:10:00", #HH:MM:SS
        "nodes_per_block": 1,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
        "available_accelerators": 4,  # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
    
    }
    config = Config(
        run_dir="/grand/projects/FRAME-IDP/shahashka/runinfo_causalbench",
        executors=[
            HighThroughputExecutor(
            #IPyParallelExecutor(
            label='causalbench',
            available_accelerators=4,  # number of GPUs
            max_workers_per_node=4,
            provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                    launcher=SingleNodeLauncher(), 
                    worker_init=user_opts['worker_init']
                ),
            cpu_affinity="block-reverse",
                )
	],
    monitoring=MonitoringHub(
        hub_address=address_by_hostname(),
        monitoring_debug=False,
        resource_monitoring_interval=10,
    ),
)
    return config


