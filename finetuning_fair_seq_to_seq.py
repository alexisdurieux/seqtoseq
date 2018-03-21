#!/usr/bin/python3
import os

LEARNING_RATES =  [0.025]
CLIP_NORMS = [0.01, 0.1]
DROPOUTS = [0.2, 0.3, 0.4]
MAX_TOKENS = [1000, 2000, 3000]  

def main():
    os.system("mkdir logs")	
    for lr in LEARNING_RATES:
        for clip_norm in CLIP_NORMS:
            for dropout in DROPOUTS:
                for max_tokens in MAX_TOKENS:
                    os.system("CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en  --lr {0} --clip-norm {1} --dropout {2} --max-tokens {3} --arch fconv_iwslt_de_en --save-dir checkpoints/fconv_{0}_{1}_{2}_{3} > logs/fconv_{0}_{1}_{2}_{3}".format(str(lr), str(clip_norm), str(dropout), str(max_tokens)))

if __name__ == '__main__':
    main()
