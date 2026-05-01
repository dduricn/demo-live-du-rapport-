# Clustering des Recettes par Similarité de Processus

## Objectif

Regrouper les recettes qui partagent des processus culinaires similaires en se basant sur leurs séquences d'actions générées par le pipeline LLM. L'idée est de découvrir des familles de recettes qui suivent le même patron de préparation, indépendamment de leur catégorie ou de leurs ingrédients.

---

## Étapes d'implémentation

### Étape 1 — Préparation des données

- Extraire les listes d'actions de la variante principale pour chaque recette depuis `data_wide`
- Nettoyer et normaliser les verbes (minuscules, suppression des doublons si nécessaire)
- Résultat : une liste de 34 591 séquences d'actions

### Étape 2 — Vectorisation des séquences

- Transformer chaque séquence d'actions en vecteur numérique avec **TF-IDF** (Term Frequency - Inverse Document Frequency)
- Chaque verbe d'action devient une dimension du vecteur
- Le poids TF-IDF donne plus d'importance aux verbes rares et spécifiques (ex. « flamber ») qu'aux verbes communs (ex. « mix », « serve »)
- Résultat : une matrice (34 591 recettes × N verbes uniques)

### Étape 3 — Réduction de dimensionnalité

- Appliquer **PCA** (Principal Component Analysis) pour réduire le nombre de dimensions tout en conservant la variance
- Réduire à ~50 composantes pour le clustering, puis à 2 composantes pour la visualisation
- Alternative : utiliser **t-SNE** ou **UMAP** pour la visualisation 2D (meilleure séparation visuelle des clusters)

### Étape 4 — Choix du nombre de clusters

- Tester plusieurs valeurs de K (ex. 3 à 15) avec **K-Means**
- Utiliser la **méthode du coude** (Elbow Method) : tracer l'inertie en fonction de K et identifier le point d'inflexion
- Calculer le **score silhouette** pour chaque K : mesure la qualité de séparation des clusters (plus proche de 1 = meilleur)
- Sélectionner le K optimal

### Étape 5 — Clustering

- Appliquer **K-Means** avec le K optimal sur les vecteurs réduits
- Assigner chaque recette à un cluster
- Alternative : tester **DBSCAN** si on suspecte des clusters de formes irrégulières ou des outliers

### Étape 6 — Analyse des clusters

Pour chaque cluster, analyser :
- **Taille** : nombre de recettes dans le cluster
- **Verbes dominants** : top 10 des verbes les plus fréquents dans le cluster
- **Catégorie culinaire** : répartition bakery / stew / quick_prep / other
- **Complexité** : répartition simple / moyenne / élevée
- **Nombre moyen d'actions** : comparer entre clusters

### Étape 7 — Visualisation (3 graphiques)

1. **Scatter plot 2D** (PCA ou t-SNE) : chaque point = une recette, coloré par cluster
2. **Heatmap** : verbes dominants par cluster (lignes = clusters, colonnes = verbes, couleur = fréquence)
3. **Barres groupées** : répartition des catégories culinaires par cluster

---

## Librairies nécessaires

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
```

---

## Résultat attendu

Identifier des groupes comme :
- Cluster A : recettes axées **mélange + cuisson au four** (pâtisseries)
- Cluster B : recettes axées **découpe + cuisson longue** (ragoûts, mijotés)
- Cluster C : recettes axées **assemblage + service rapide** (salades, sandwichs)
- etc.

Cela permet de valider que le pipeline génère des séquences d'actions cohérentes et que les processus culinaires émergent naturellement des données.
