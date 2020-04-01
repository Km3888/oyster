#!/bin/sh
#
#SBATCH --verbose
#SBATCH -p gpu
#SBATCH --job-name=pearl
#SBATCH --output=pearl%j.out
#SBATCH --error=pearl%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=km3888@nyu.edu

module load anaconda3
module load cuda/9.0
source activate rl


python /gpfsnyu/home/km3888/oyster/launch_experiment.py /gpfsnyu/home/km3888/oyster/configs/cheetah-dir.json
