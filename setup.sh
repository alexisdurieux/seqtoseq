# Installation
mkdir project &&
cd project/ &&
git clone https://github.com/facebookresearch/fairseq-py.git &&
cd fairseq-py/ &&
virtualenv venv --python=python3 &&
source venv/bin/activate &&
pip install -r requirements.txt &&
python setup.py build &&
python setup.py develop &&
pip install pytorch &&

# Preprocessing
cd data/ &&
bash prepare-iwslt14.sh &&
cd .. &&
python preprocess.py --source-lang de --target-lang en   --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test   --destdir data-bin/iwslt14.tokenized.de-en &&
mkdir -p checkpoints/fconv &&

# Training 
CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en   --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000   --arch fconv_iwslt_de_en --save-dir checkpoints/fconv &&

# Generation
python generate.py data-bin/iwslt14.tokenized.de-en   --path checkpoints/fconv/checkpoint_best.pt   --batch-size 128 --beam 5 &&
