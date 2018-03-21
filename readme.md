# Projet Deep Learning - Master SID

## FAIR SEQ2SEQ Experiments
Facebook a développé une librairie afin d'effecture d'effectuer du **sequence to sequence learning**. Cette implémentation permet à l'aide d'un CLI fourni d'effectuer des apprentissages de A à Z en passant du preprocessing au scoring. Afin de faciliter l'utilisation et l 'apprentissage de ces modèles j'ai regroupé un ensemble de commande permettant de mettre en place le projet ainsi que de lancer un apprentissage. Ces commandes sont listées dans l'encadré de code ci-dessous. De plus le fishier `shell` *interactive_generation.sh* permet de tester les résultats de l'apprentissage avec une ligne de commande interactive. Enfin le fichier `finetuning_fair_seq_to_seq.py`, est un script permettant de lancer plusieurs apprentissages à la suite sur une série d'hyperparamètres différents à des fins de fine-tuning du modèle. Néanmoins, en raison d'un espace disque insuffisant sur les serveurs de l'université mis à notre disposition, il m'a été impossible de comparer plus de 3 modèles.  simultanément. L'idée à la base était d'effectuer une série d'apprentissage et de stocker les logs dans des fichiers. Puis à l'aide d'un parsing de log de sélectionner le modèle dont les hyper paramètres offraient les meilleurs résultats.

```
mkdir project
cd project/
git clone https://github.com/facebookresearch/fairseq-py.git
cd fairseq-py/ 
virtualenv venv --python=python3
source venv/bin/activate 
pip install wheel
pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
pip install -r requirements.txt 
python setup.py build 
python setup.py develop
pip install pytorch 

# Preprocessing
cd data/ 
bash prepare-iwslt14.sh
cd ..
python preprocess.py --source-lang de --target-lang en   --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test   --destdir data-bin/iwslt14.tokenized.de-en 
mkdir -p checkpoints/fconv 

# Training 
CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en   --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000   --arch fconv_iwslt_de_en --save-dir checkpoints/fconv 

# Generation
python generate.py data-bin/iwslt14.tokenized.de-en   --path checkpoints/fconv/checkpoint_best.pt   --batch-size 128 --beam 5

```

## Convolutional SEQ2SEQ implementation with Keras

De plus j'ai essayé d'implémenter un réseau convolutionnel Seq2Seq avec Keras. Afin de faciliter l'implémentation, j'ai utilisé un jeu de données simplifié afin d'avoir moins de preprocessing à faire. Le jeu de données consiste en une liste de phrases anglaises séparées de leurs traductions françaises par un caractère `\t`. À partir de ce jeu de données, j'ai dans un premier temps créé le dictionnaire de mots source (anglais) et de mots de sortie (anglais) afin de créer l'embedding des données. De plus en raison du caractère convolutionel du réseau et donc de la modélisation hierarchique des données, il est précisé dans le papier l'importance de donner la position des mots en plus de leur présence ou non comme précisé dans le papier. Néanmoins l'implémentation convolutionelle n'a pas abouti. De plus afin de réaliser l'implémentation équivalent à celle du papier, il faudrait développer un `Layer` keras `AttentionLayer` permettant de reproduire le modèle à attention du papier.

## MLP Implémentation

Enfin, j'ai réalisé dans le notebook MLP from Scratch, une implémentation du concept de layers théorisé par Y.Le Cun. Cette implémentation m'a permis de reproduire l'algorithme de *retropropagation* et de comprendre le concept de layers dans un réseau de neurones.

## Synthèse

La synthèse du papier est disponible dans le fichier `synthèse.pdf` à la source de ce projet