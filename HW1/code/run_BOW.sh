#!/usr/bin/env bash
module load cuda/9.0.176
module load cudnn/9.0v7.0.5

source activate py36_inex

export NAME=$SLURM_JOB_NAME
#INPUTDIR=df07f20K_stopEng_C_${SLURM_ARRAY_TASK_ID}gram
#INPUTDIR=df07f50K_stopEng_W_1_3gram
INPUTDIR=df07f20K_stopEng_W_1gram
RUNDIR=/scratch/jl7722/project/NLP_HW1/model/${INPUTDIR}/$NAME
mkdir -p $RUNDIR
cd $RUNDIR

#name='doc300Vocab20K_AdamLR0.01ExpDecay_embdim50'
#K=1
NAME+="_"
NAME+=$SLURM_ARRAY_TASK_ID
#name+=$K

#python -u /scratch/jl7722/project/NLP_HW1/code/train.py --doc_len 350 --dimEmb 50 --n_iter 100 --flgSave --lr_decay_type exp --lr_decay3 10 --lr 0.01 --batchSize 256 --n_batch 5 --nVocab 50000 --savePath /scratch/jl7722/project/NLP_HW1/model/${INPUTDIR}/${NAME}/ --inputPath /scratch/jl7722/project/NLP_HW1/aclImdb/${INPUTDIR}/ --flg_cuda > /scratch/jl7722/project/NLP_HW1/model/${INPUTDIR}/${NAME}_log.txt
#2>&1 &
#
#--optType SGD
#--flgSave
# --dimEmb 100
# --i $SLURM_ARRAY_TASK_ID

python -u /scratch/jl7722/project/NLP_HW1/code/train.py --i $SLURM_ARRAY_TASK_ID --doc_len 300 --dimEmb 100 --n_iter 100 --flgSave --lr 0.01 --batchSize 256 --n_batch 5 --nVocab 50000 --savePath /scratch/jl7722/project/NLP_HW1/model/${INPUTDIR}/${NAME}/ --inputPath /scratch/jl7722/project/NLP_HW1/aclImdb/${INPUTDIR}/ --flg_cuda > /scratch/jl7722/project/NLP_HW1/model/${INPUTDIR}/${NAME}_log.txt
