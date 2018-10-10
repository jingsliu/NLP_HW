#!/usr/bin/env bash
export JOB_NAME=$1
shift
export DL_ARGS=$@

echo $JOB_NAME
echo $DL_ARGS

sbatch -J $JOB_NAME --nodes=1 --tasks-per-node=1 -t00:30:00 --mem=5GB --gres=gpu:1 --array=1-2 --output=out/$JOB_NAME.o_%A_%a run_BOW.sh
