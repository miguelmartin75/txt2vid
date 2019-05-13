#!/bin/bash

# Configure the resources required
#SBATCH -p batch                                                # partition (this is the queue your job will be added to)
#SBATCH -n 1              	                                # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 8              	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=2-00:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:kepler:1                                            # generic resource required (here requires 4 GPUs)
#SBATCH --mem=64GB 

# Configure notifications 
#SBATCH --mail-type=END                                         # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
#SBATCH --mail-type=FAIL                                        # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
#SBATCH --mail-user=a1669396@student.adelaide.edu.au                    # Email to which notification will be sent

# Execute your script (due to sequential nature, please select proper compiler as your script corresponds to)
module load Python/3.6.1-foss-2016b
module load cuDNN/7.3.1-CUDA-9.0.176
source $FASTDIR/txt2vid/env/bin/activate

export PYTHONPATH="./:$PYTHONPATH"
bash ./run.sh

deactivate
