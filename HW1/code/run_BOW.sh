#!/usr/bin/env bash
module load cuda/9.0.176
module load cudnn/9.0v7.0.5

source activate py36_inex

export NAME=$SLURM_JOB_NAME
RUNDIR=/scratch/jl7722/project/NLP_HW1/model/df07f20K_stopEng_W_1gram/$NAME
mkdir -p $RUNDIR
cd $RUNDIR

#name='doc300Emb100Vocab20K_AdamLrdecayNone_testLR'
#K=1
NAME+="_"
NAME+=$SLURM_ARRAY_TASK_ID
#name+=$K

python -u /scratch/jl7722/project/NLP_HW1/code/train.py --doc_len 300 --n_iter 100 --i $SLURM_ARRAY_TASK_ID --batchSize 256 --n_batch 5 --lr_decay_type None --dimEmb 100 --nVocab 20000 --flgSave --savePath /scratch/jl7722/project/NLP_HW1/model/df07f20K_stopEng_W_1gram/${NAME}/ --inputPath /scratch/jl7722/project/NLP_HW1/aclImdb/df07f20K_stopEng_W_1gram/ --flg_cuda > /scratch/jl7722/project/NLP_HW1/model/df07f20K_stopEng_W_1gram/${NAME}_log.txt
#2>&1 &