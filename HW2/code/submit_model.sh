#!/usr/bin/env bash
export JOB_NAME=$1
shift
export DL_ARGS=$@

echo $JOB_NAME
echo $DL_ARGS

sbatch -J $JOB_NAME --nodes=1 --tasks-per-node=1 -t01:00:00 --mem=5GB --gres=gpu:1 --array=1 --output=out/$JOB_NAME.o_%A_%a run_CNN.sh

#sbatch -J $JOB_NAME --nodes=1 --tasks-per-node=1 -t01:00:00 --mem=5GB --gres=gpu:1 --array=3 --output=out/$JOB_NAME.o_%A_%a run_RNN.sh
