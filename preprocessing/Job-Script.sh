#!/bin/bash

# This script loops through all subject IDs and runs fmriprep in parallel
# Change the SBATCH arguments as you see fit
#
# Ian Richard Ferguson | Stanford University

project_directory="${SCRATCH}/SCP"
mkdir -p $project_directory && cd $project_directory

job_directory="${project_directory}/.job"
out_directory="${project_directory}/.out"

mkdir -p $job_directory && mkdir -p $out_directory

data_dir="/oak/stanford/groups/jzaki/scp_2022/bids/"

subjects=('11687' '10724' '10617')

for sub in ${subjects[@]}; do

    job_file="${job_directory}/${sub}.job"

    echo "#!/bin/bash
#SBATCH --job-name=${sub}.job
#SBATCH --output=.out/${sub}.out
#SBATCH --error=.out/${sub}.err
#SBATCH --time=2-00:00
#SBATCH --mem=24000
#SBATCH --qos=normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$USER@stanford.edu
#SBATCH -c 10
#SBATCH -N 1
bash /oak/stanford/groups/jzaki/scp_2022/scripts/preprocessing/fmriprep_singleSubject.sh $sub" > $job_file
    sbatch $job_file

done
