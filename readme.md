# Projet Deep Learning - Master SID

## FAIR SEQ2SEQ Experiments
Facebook a développé une librairie afin d'effecture d'effectuer du **sequence to sequence learning**. Cette implémentation permet à l'iade d'un CLI fourni d'effectuer des apprentissages de A à Z en passant du preprocessing au scoring. Afin de faciliter l'utilisation et l 'apprentissage de ces modèles j'ai regroupé dans des fichiers *shell*, `preprocessing.sh`, `learning.sh` un ensemble de commande permettant d'effectuer un apprentissage rapidement sur la librairie. De plus le fichier `multi_learning.py`, est un script permettant de lancer plusieurs apprentissages à la suite sur une série d'hyperparamètres différents à des fins de fine-tuning du modèle. Néanmoins, en raison d'un espace disque insuffisant sur les serveurs de l'université mis à notre disposition, il m'a été impossible de comparer plus de 3 modèles simultanément.

## Convolutional SEQ2SEQ implementation with Keras
