#!/usr/bin/env bash
source activate py36_inex

python -u ./eval.py --doc_len 350 --modelPath ../model/df07f50K_stopEng_W_1_3gram/doc350Vocab50K13Gram_AdamLR0.01ExpDecay_embdim50Save_1/ --inputPath ../aclImdb/df07f50K_stopEng_W_1_3gram/ > ../model/df07f50K_stopEng_W_1_3gram/doc350Vocab50K13Gram_AdamLR0.01ExpDecay_embdim50Save_1/testresult.txt