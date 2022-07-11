#!/bin/bash
SBATCH --job-name=fit.job
SBATCH --output=fit.out
SBATCH --error=fit.err
SBATCH --time=2-00:00
SBATCH --mem=24000
SBATCH --qos=normal
SBATCH --mail-type=FAIL,END
SBATCH --mail-user=$USER@stanford.edu
SBATCH --c 10
SBATCH -N 1

ml python/3.6.1
python3 fit_matrices.py $1