#!/usr/bin/env bash
source activate py36_inex
#name='df07f20K_stopEng_C_3gram'
name='df07f100K_stopEng_W_1gram'
mkdir ../aclImdb/${name}
#python -u ./tokenization.py --outPath ../aclImdb/df07f20K_stopEng_W_1gramV2/
#python -u ./tokenization.py --outPath ../aclImdb/df07f20K_stopEng_W_2gram/ --ngramMax 2 --ngramMin 2
#python -u ./tokenization.py --outPath ../aclImdb/${name}/ --ngramMax 3 --ngramMin 3 > ../${name}/log.txt
python -u ./tokenization.py --outPath ../aclImdb/${name}/ --ngramMax 1 --ngramMin 1 --max_features 100000 > ../aclImdb/${name}/log.txt

#python -u ./tokenization.py --max_df 0.7 --analyzer char --outPath ../aclImdb/${name}/ --ngramMax 3 --ngramMin 3 --max_features 20000 > ../aclImdb/${name}/log.txt

