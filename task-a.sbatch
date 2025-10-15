#!/bin/bash
### Some common partitions
##SBATCH --partition=gpu_a100
##SBATCH --partition=sooner_gpu_test_ada
##SBATCH --partition=sooner_gpu_test
#SBATCH --partition=gpu
##SBATCH --partition=debug_gpu # This partition is currently selected.
#
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=20
##SBATCH --mem=32G
#SBATCH --mem=16G
##SBATCH --time=12:00:00
##SBATCH --time=00:20:00
#SBATCH --time=12:00:00
#SBATCH --job-name=exp1
#SBATCH --mail-user=radmanesh@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs529329/CodeSherlock
#SBATCH --output=results/exp1_%j_stdout.txt
#SBATCH --error=results/exp1_%j_stderr.txt
##SBATCH --array=0-4    # the double ## means that this line is ignored

#################################################

# NOTE, make sure you change your directory!

MM="/scratch/cs529329/micromamba/bin/micromamba"

eval "$("$MM" shell hook -s bash)"

micromamba activate nlpproject

python train-baseline.py
