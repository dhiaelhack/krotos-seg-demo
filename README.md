# Démo : segmentation de la vapeur dans des images radioscopiques (données synthétiques)

Auteur : Dhia Elhack NSIBI (ENISo, Tunisie) – démo préparée en lien avec le sujet de stage
« Deep learning / computer vision pour l'analyse d'images radioscopiques d'un mélange triphasique » (LMAG).

## Ce que c'est, et ce que ce n'est pas

C'est une petite chaîne complète **débruitage → segmentation en 3 classes (eau / corium / vapeur) → évaluation**,
exécutée sur des images **simulées**. Elle sert à montrer ma manière de travailler : construire un banc
d'essai, mesurer avec des métriques, comparer des méthodes.

**Ce n'est pas** un résultat sur des images KROTOS. Je n'ai aucune donnée réelle, et les chiffres ci-dessous
ne disent rien de la performance sur de vraies images. Les images sont générées par `make_image` :
jet de corium sinueux et gouttes (sombres), bulles de vapeur (peu contrastées, bords flous), fond d'eau non
uniforme, flou optique et bruit gaussien.

## Ce que fait le code (`demo.py`)

- **Débruitage** : aucun, gaussien, médian, non-local means (NLM).
- **Segmentation sans apprentissage** : seuillage d'Otsu à 3 classes, k-means (k=3), k-means avec contexte spatial.
- **Segmentation supervisée** : forêt aléatoire sur des caractéristiques multi-échelles par pixel
  (intensités lissées, fond corrigé, gradient). Entraînement et test sur des **images différentes**.
- **Évaluation** : coefficient de Dice par classe, moyenné sur 10 images de test, pour trois niveaux de bruit.

## Résultats (σ = 0,04 ; Dice moyen de la classe vapeur ; 10 images de test, 6 d'entraînement)

| Méthode | Sans débruitage | Avec NLM |
|---|---|---|
| Otsu, 3 classes | 0,31 | 0,36 |
| k-means, k=3 | – | 0,39 |
| Forêt aléatoire (supervisé) | 0,79 | 0,82 |

Les résultats complets sont dans `results/resultats.csv`, les figures dans `results/`.

## Ce que j'en retiens (sur ces données simulées)

1. La **vapeur est la classe difficile** : son contraste avec l'eau est faible et le fond n'est pas uniforme,
   alors que le corium (très sombre) est presque toujours bien segmenté (Dice > 0,94).
2. **Le débruitage seul aide peu** les méthodes par seuil global : le facteur limitant est le contraste et le fond,
   pas le bruit.
3. Une méthode **supervisée à caractéristiques multi-échelles** améliore nettement la vapeur ; la comparaison
   avec un vrai réseau de segmentation reste à faire.
4. Limites : données simulées, forêt aléatoire et non réseau profond, une seule configuration de bruit gaussien.

## Suite que je proposerais sur de vraies images

- Reprendre les travaux existants (SPECTRA) comme référence et comparer à leurs résultats.
- Modèle de segmentation (type U-Net, PyTorch) avec perte de Dice ou focale, et masquage du corium.
- Validation par essai : entraînement sur un essai KROTOS, test sur un autre.
- Métriques adaptées à l'usage : Dice et IoU, mais aussi l'erreur sur la fraction de vapeur.

## Utilisation

```bash
pip install -r requirements.txt
python demo.py --n 10 --seed 0     # environ 4 minutes sur un seul cœur
```
