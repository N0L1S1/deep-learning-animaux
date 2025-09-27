# Deep Learning - Classification Animaux (Plumes vs Poils)

## Description
Projet réalisé en 3e année Réseaux & Télécoms.  
Implémentation d’un neurone logistique en Python pour prédire si un animal possède **des plumes (1)** ou **des poils (0)** à partir de ses caractéristiques (taille, poids, vole, pond des oeufs).

## Contenu du dépôt
- `final.py` : script Python principal  
- `animaux_classification.csv` : dataset utilisé pour l’entraînement  
- `logloss_curve.png` : courbe de la fonction de coût (log loss)  

## Installation des dépendances
Avant d’exécuter le projet, installer les bibliothèques nécessaires :

```bash
pip install numpy
pip install pandas
pip install matplotlib
```

## Mathématiques du modèle
Le neurone logistique fonctionne ainsi :

- Combinaison linéaire des entrées :
z = w1 * x1 + w2 * x2 + ... + wn*xn + b

- Activation sigmoïde :
a(z) = 1 / (1 + exp(-z))
(interprétée comme probabilité P(y=1|x))

- Fonction de coût (log loss) :
L(a,y) = -( y*log(a) + (1-y)*log(1-a) )

- Descente de gradient :
wi <- wi - alpha * dL/dwi

b <- b - alpha * dL/db

où alpha est le learning rate.


## Résultats attendus
- Affichage en console de la progression de l’entraînement (loss, accuracy, poids du modèle).

- Génération de la courbe logloss_curve.png.

- Quelques exemples de prédiction sur des animaux typiques (oiseau, mammifère, ornithorynque).

