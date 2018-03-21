# Projet Deep Learning - Master SID

## FAIR SEQ2SEQ Experiments
Facebook a développé une librairie afin d'effecture d'effectuer du **sequence to sequence learning**. Cette implémentation permet à l'iade d'un CLI fourni d'effectuer des apprentissages de A à Z en passant du preprocessing au scoring. Afin de faciliter l'utilisation et l 'apprentissage de ces modèles j'ai regroupé dans des fichiers *shell*, `setup.sh`, `interactive_generation.sh` un ensemble de commande permettant d'effectuer un apprentissage rapidement sur la librairie et de voir les résultats en temps réel. De plus le fichier `finetuning_fair_seq_to_seq.py`, est un script permettant de lancer plusieurs apprentissages à la suite sur une série d'hyperparamètres différents à des fins de fine-tuning du modèle. Néanmoins, en raison d'un espace disque insuffisant sur les serveurs de l'université mis à notre disposition, il m'a été impossible de comparer plus de 3 modèles.  simultanément. L'idée à la base était d'effectuer une série d'apprentissage et de stocker les logs dans des fichiers. Puis à l'aide d'un parsing de log de sélectionner le modèle dont les hyper paramètres offraient les meilleurs résultats.

```
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
```
# Generation
python generate.py data-bin/iwslt14.tokenized.de-en   --path checkpoints/fconv/checkpoint_best.pt   --batch-size 128 --beam 5

## Convolutional SEQ2SEQ implementation with Keras
