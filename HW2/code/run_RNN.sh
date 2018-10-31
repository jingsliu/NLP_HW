#!/usr/bin/env bash
module load cuda/9.0.176
module load cudnn/9.0v7.0.5

source activate py36_inex

export NAME=$SLURM_JOB_NAME
INPUTDIR=df07f20K
TYPE=RNN
NAME+="_"
NAME+=$SLURM_ARRAY_TASK_ID
RUNDIR=/scratch/jl7722/project/NLP_HW1/model/${TYPE}/$NAME
mkdir -p $RUNDIR
cd $RUNDIR

python -u /scratch/jl7722/project/NLP_HW2/code/train.py --flgSave --flgProd --doc_len 30 --i $SLURM_ARRAY_TASK_ID --n_iter 150 --n_batch 100 --lr 1e-2 --modelName RNN --batchSize 64 --flg_cuda --inputPath /scratch/jl7722/project/NLP_HW2/hw2_data/${INPUTDIR}/ --savePath /scratch/jl7722/project/NLP_HW2/model/${TYPE}/${NAME}/ > /scratch/jl7722/project/NLP_HW2/model/${TYPE}/${NAME}_log.txt

#--flgSave
#
#--K1 5 --K2 3
# --filters1 256 --filters2 256 --L1 128


#python -u /scratch/jl7722/project/NLP_HW2/code/train.py --doc_len 30 --i 3 --n_iter 2 --n_batch 10 --lr 1e-3 --modelName CNN --batchSize 64 --filters1 64 --filters2 64 --L1 32 --flg_cuda --inputPath /scratch/jl7722/project/NLP_HW2/hw2_data/df07f20K/
