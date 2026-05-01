# Reconnaissance d'activités culinaires — Rapport Stage Projet I

Démo Streamlit du projet de reconnaissance de recettes par séquence de gestes.

## Contenu

Présentation interactive du pipeline complet :

1. **Clustering** — Regroupement des 34 591 recettes par similarité de gestes (TF-IDF + PCA + K-Means avec K=25)
2. **Méthode 1** — Trie + scoring multi-critères avec pré-filtre cluster
3. **Méthode 2** — Classification hiérarchique (Decision Tree + Random Forest) pour prédire le cluster
4. **Comparaison finale** — Synthèse des résultats des 3 systèmes

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement

Application déployée via [Streamlit Community Cloud](https://streamlit.io/cloud).

## Auteur

David Ndimina — Stage Projet I, encadré par Bruno Bouchard (LIARA)
