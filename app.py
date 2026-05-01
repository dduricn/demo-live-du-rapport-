"""
Streamlit — Rapport 2 du stage Projet I 
Auteur : David
Date   : 2026-04-23

Lancement :
    streamlit run app.py
    conda activate jupyter_clean
   streamlit run app.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import ast

# ─────────────────────────────────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stage Projet I - reconnaissance des activités - LIARA",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Style CSS minimal
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C3E50;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3498DB;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #34495E;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498DB;
    }
    /* Largeur fixe de la sidebar quand elle est ouverte
       (cible le div interne pour ne pas casser l'animation de fermeture) */
    section[data-testid="stSidebar"] > div:first-child {
        width: 300px;
        min-width: 300px;
        max-width: 300px;
    }
    /* Quand sidebar fermee : prend toute la largeur */
    section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 0;
        min-width: 0;
        max-width: 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — navigation
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.title("Informations")
st.sidebar.markdown("**Titre :** : Stage Projet I")
st.sidebar.markdown("**Nom-etudiant:** : David Ndimina")
st.sidebar.markdown("**Enseignant:** : Bruno Bouchard")
st.sidebar.title("Données")
st.sidebar.markdown("**Dataset** : 34 591 recettes")
st.sidebar.markdown("**Clusters** : 25 (K-Means)")
st.sidebar.title("Navigation")

# Liste des pages dans l'ordre
PAGES = [
    "Accueil",
    "Clustering",
    "Methode 1 — Trie + Scoring",
    "Methode 2 — Classification",
    "Comparaison finale",
]

# Initialisation : page courante stockee dans session_state
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

# Le radio sidebar utilise directement session_state via key="page"
st.sidebar.radio("Aller à :", PAGES, key="page")

# Fonctions de navigation (utilisees par les boutons fleches)
def go_prev():
    idx = PAGES.index(st.session_state.page)
    if idx > 0:
        st.session_state.page = PAGES[idx - 1]

def go_next():
    idx = PAGES.index(st.session_state.page)
    if idx < len(PAGES) - 1:
        st.session_state.page = PAGES[idx + 1]

# Variable utilisee dans les if/elif ci-dessous
page = st.session_state.page

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 : ACCUEIL
# ─────────────────────────────────────────────────────────────────────────────
if page == "Accueil":
    st.markdown('<div class="main-title">Reconnaissance d&apos;activté</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Travail réalisé
        Aprés l'exploration et le passage du dataset_flags_critique dans les llms, nous avons travaillé  avec le datase_wide qui est la 
        composition de chaque ID de recette lié à ses variantes(principale, ingredient, permutation)
    ### dans un pipeline de 3 étapes
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Clustering")
        st.markdown("""
        - TF-IDF sur les verbes
        - PCA à 50 dimensions
        - K-Means avec K=25
        """)
    with col2:
        st.markdown("#### 2. Option 1 — Trie + Scoring")
        st.markdown("""
        - Trie sur les séquences
        - Scoring multi-critères
        - Filtre par cluster
        """)
    with col3:
        st.markdown("#### 3. Option 2 — Classification")
        st.markdown("""
        - DTC + Random Forest
        - Prédiction hiérarchique du cluster
        - Intégration avec le Trie
        """)
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 : CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Clustering":
    st.markdown('<div class="main-title">Clustering des recettes</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Objectif
    Regrouper les **34 591 recettes** en familles culinaires similaires, en se basant
    uniquement sur la **séquence de gestes** , les verbes d'actions, sans utiliser le titre,
    la catégorie ou les ingrédients.
    """)

    st.markdown("---")
    st.markdown("## Pipeline en 3 étapes")

    # ─── Étape 1 : TF-IDF
    st.markdown("### 1. TF-IDF — Vectorisation des recettes")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **TF-IDF** = *Term Frequency × Inverse Document Frequency*

        Une méthode pour transformer chaque recette en **vecteur numérique**,
        où les **verbes rares et spécifiques** ont plus de poids que les verbes communs.

        Les verbes comme `chop`, `mix`, `serve` apparaissent partout, dant toutes les recettes. ils
        n'aident pas à distinguer les recettes. Les verbes comme `knead`  ou
        `caramelize` sont rares et **discriminants** : ils révèlent le type de recette.
        """)
    with col2:
        st.info("""
        **Exemple**

        Recette : `[chop, mix, knead, bake]`

        | Verbe | Poids TF-IDF |
        |---|---|
        | `chop` | 0.18 (commun) |
        | `mix` | 0.15 (commun) |
        | `knead` | **0.85** (rare) |
        | `bake` | 0.45 (moyen) |
        """)

    st.markdown("**Résultat** : chaque recette devient un vecteur de **344 dimensions** (un nombre par verbe distinct du vocabulaire).")

    st.markdown("---")

    # ─── Étape 2 : PCA
    st.markdown("### 2. PCA — Réduction de dimensionnalité")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **PCA** = *Principal Component Analysis* (Analyse en Composantes Principales)

        Une technique qui permet de reduire le nombre de dimensions tout en gardant l'essentiel
        de l'information.

        En haute dimension 344(dimension obtenu avec la verctorisation des recettes), les distances entre les points deviennent
        toutes similaires,  K-Means ne peut plus regrouper
        correctement. PCA trouve les **axes où les données varient le plus** et garde
        seulement ceux-là.
        """)
    with col2:
        st.info("""
        **Réduction**

        Avant : `34 591 × 344`
        Après : `34 591 × 50`

        **Variance conservée** : 70.8%

        → Calculs plus rapides, K-Means plus efficace, moins de bruit
        """)

    st.markdown("---")

    # ─── Étape 3 : K-Means
    st.markdown("### 3. K-Means — Regroupement en 25 clusters")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **K-Means** : algorithme qui regroupe les données en **K groupes** (clusters)
        en plaçant K centres et en assignant chaque recette à son centre le plus proche.

        Le Choix de k=25 a été fait via la **méthode du coude** et le **score silhouette** :
        K=25 offre des clusters **suffisamment fins** pour distinguer les familles culinaires
        (salades vs soupes vs pâtisseries) sans être trop granulaire.
        """)
    with col2:
        st.success("""
        **Algorithme**

        1. Placer 25 centres
        2. Chaque recette → centre le plus proche
        3. Recalculer chaque centre
        4. Répéter jusqu'à convergence
        """)

    st.markdown("---")

    # ─── Exemples de clusters obtenus
    st.markdown("## Exemples de clusters obtenus")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### Cluster 13 — Pâtisseries")
        st.markdown("""
        **Verbes dominants** :
        `sift, beat, bake, chill, knead, roll`

        **Exemples** :
        - Pfeffernusse (Pepper Balls)
        - Mash Potato Puffs
        - Cookies divers
        """)
    with col2:
        st.markdown("#### Cluster 14 — Cuisson rapide")
        st.markdown("""
        **Verbes dominants** :
        `mix, heat, bake, sprinkle`

        **Exemples** :
        - Buttercream Frosting
        - Oven Dried Tomatoes
        - Gardenburger Original Veggie Patty (Copycat)
        """)
    with col3:
        st.markdown("#### Cluster 20 — Salades")
        st.markdown("""
        **Verbes dominants** :
        `chop, mix, season, serve`

        **Exemples** :
        - Black Bean Salad
        - Roasted Corn Guacamole
        - Bean Salad
        """)
    with col4:
        st.markdown("#### Cluster 24 — Viandes")
        st.markdown("""
        **Verbes dominants** :
        `dredge, brown, simmer, skim`

        **Exemples** :
        - Wine Sauced Chicken
        - Veal Stew Marengo
        - Plats mijotés
        """)

    st.markdown("---")
    st.markdown("## Visualisation 2D des 25 clusters")

    # Couleurs HEX exactement identiques a celles du notebook (tab20 + tab20b[:5])
    cluster_hex = {
        0:  "#1f77b4", 1:  "#aec7e8", 2:  "#ff7f0e", 3:  "#ffbb78",
        4:  "#2ca02c", 5:  "#98df8a", 6:  "#d62728", 7:  "#ff9896",
        8:  "#9467bd", 9:  "#c5b0d5", 10: "#8c564b", 11: "#c49c94",
        12: "#e377c2", 13: "#f7b6d2", 14: "#7f7f7f", 15: "#c7c7c7",
        16: "#bcbd22", 17: "#dbdb8d", 18: "#17becf", 19: "#9edae5",
        20: "#393b79", 21: "#5254a3", 22: "#6b6ecf", 23: "#9c9ede",
        24: "#637939",
    }

    # Descriptions connues (clusters majeurs)
    cluster_desc = {
        13: "Pâtisseries",
        14: "Cuisson rapide",
        20: "Salades",
        24: "Viandes",
    }

    col_img, col_legend = st.columns([3, 1])

    with col_img:
        st.image("img/clustering_pca_2d.png",
                 caption="Projection PCA 2D des 34 591 recettes — chaque couleur = un cluster")

    with col_legend:
        st.markdown("**Légende**")
        legend_html = "<div style='font-size: 0.85rem; line-height: 1.6;'>"
        for c in range(25):
            color = cluster_hex[c]
            desc = cluster_desc.get(c, "")
            label = f"<b>Cluster {c}</b>"
            if desc:
                label += f" — {desc}"
            legend_html += (
                f"<div><span style='display:inline-block; width:12px; height:12px; "
                f"background-color:{color}; border-radius:50%; margin-right:6px; "
                f"vertical-align: middle;'></span>{label}</div>"
            )
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 : STRATÉGIE 1
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Methode 1 — Trie + Scoring":
    st.markdown('<div class="main-title">Methode 1 — Trie + Scoring multi-critères</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Objectif
    À partir d'une **séquence partielle de gestes observés** (ex : `[chop, mix, heat]`), la methode du trie permettra 
    une identification rapide des recettes les plus probables parmi les **34 591 recettes** du dataset.
    """)

    st.markdown("---")
    st.markdown("## Pipeline en 3 étapes")

    # ─── Étape 1 : Trie
    st.markdown("### 1. Trie — Filtrage rapide par préfixe")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        La methode de ***trie*** est une methode de filtrage d'arbre par prefixe.

        Une structure de données qui permet de retrouver rapidement
        toutes les recettes qui commencent par une séquence donnée de gestes.

        Chaque nœud de l'arbre représente un verbe.
        Pour aller d'un nœud à l'autre, on suit les arêtes selon les verbes
        observés. Tous les chemins partagent un préfixe commun.

        **Construction** : on insère les **3 variantes** de chaque recette
        (principale, ingrédients, permutation) → une recette est trouvée si
        **n'importe laquelle** de ses 3 variantes correspond.
        """)
    with col2:
        st.info("""
        **Structure du Trie**

        - **103 773** séquences insérées
        (3 variantes × 34 591 recettes)

        - **441 verbes distincts** à la racine

        - Recherche en **O(m)** où m = longueur de la séquence observée
        """)

    st.markdown("**Résultat** : pour `[chop, mix, heat]`, le Trie renvoie en quelques millisecondes toutes les recettes compatibles avec ce préfixe.")

    st.markdown("---")

    # ─── Étape 2 : Scoring multi-critères
    st.markdown("### 2. Scoring multi-critères — Classement des candidats")

    st.markdown("""
    Le Trie renvoie souvent **plusieurs centaines de recettes candidates**.
    Pour les classer, on calcule un **score** pour chaque candidate à partir de **3 critères** :
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Coverage (50%)")
        st.markdown("""
        **Quelle proportion des gestes observés est présente dans la recette ?**

        Exemple :
        - Observé : `[chop, mix, heat]`
        - Recette : `[chop, mix, bake, serve]`
        - Coverage = 2/3 = 0.67

        Plus c'est haut, plus la recette **contient les bons gestes**.
        """)

    with col2:
        st.markdown("#### Position (30%)")
        st.markdown("""
        **Les gestes sont-ils dans le bon ordre ?**

        Calculé via **LCS** (Longest Common Subsequence).

        Exemple :
        - Observé : `[chop, mix, heat]`
        - Recette : `[mix, chop, heat]`
        - LCS = 2/3 = 0.67

        Plus c'est haut, plus l'**ordre est respecté**.
        """)

    with col3:
        st.markdown("#### Completion (20%)")
        st.markdown("""
        **Quelle fraction de la recette a été accomplie ?**

        Exemple :
        - Observé : `[chop, mix]`
        - Recette : 5 étapes au total
        - Completion = 2/5 = 0.40

        Plus c'est haut, plus la recette est **proche de la fin**.
        """)

    st.markdown("""
    **Score final** = `0.5 × Coverage + 0.3 × Position + 0.2 × Completion`

    Les **3 variantes** de la recette sont scorées séparément ; on garde le **meilleur score**
    parmi les trois pour assurer la robustesse.
    """)

    st.markdown("---")

    # ─── Étape 3 : Filtre cluster
    st.markdown("### 3. Filtre cluster — Réduction de l'espace de recherche")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        Avant même d'appliquer le Trie sur tout le dataset, on **réduit la recherche**
        à un seul cluster (sur les 25 obtenus par K-Means).

        **Comment ?** Pour la phase d'évaluation, on utilise le **vrai cluster**
        de la recette comme oracle (en production, ce cluster sera prédit par
        Random Forest — voir Méthode 2).

        **Bénéfices attendus :**
        - **Précision** : moins de candidats parasites
        - **Vitesse** : 25× moins de recettes à scorer
        """)
    with col2:
        st.success("""
        **Pipeline complet**

        1. Séquence observée
        2. Filtre par cluster
        3. Trie sur le cluster
        4. Scoring multi-critères
        5. Top-5 recettes
        """)

    st.markdown("---")

    # ─── Résultats
    st.markdown("## Résultats")

    st.markdown("### Précision selon la longueur d'observation (sans filtre)")

    st.markdown("""
    Test sur **200 recettes tirées au hasard**, avec différentes longueurs d'observation
    (3, 4, 5, 6 gestes).
    """)

    perf_data = pd.DataFrame({
        "Gestes observés": [3, 4, 5, 6],
        "N valides": [200, 193, 168, 110],
        "Top-1": ["13.0%", "26.4%", "39.3%", "67.3%"],
        "Top-5": ["27.5%", "43.0%", "69.6%", "87.3%"],
        "Score moyen Top-1": [0.9761, 0.9746, 0.9759, 0.9778],
    })
    st.dataframe(perf_data, hide_index=True, use_container_width=True)

    st.image("img/strategie1_performance.png",
             caption="Évolution de la précision selon le nombre de gestes observés")

    st.markdown("---")

    st.markdown("### Comparaison avec/sans filtre cluster")

    st.markdown("""
    Quand on **ajoute le filtre cluster** (oracle), la précision augmente significativement
    et le temps de réponse chute.
    """)

    cluster_data = pd.DataFrame({
        "Gestes": [3, 3, 4, 4, 5, 5, 6, 6],
        "Mode": ["Sans filtre", "Avec filtre", "Sans filtre", "Avec filtre",
                 "Sans filtre", "Avec filtre", "Sans filtre", "Avec filtre"],
        "Top-1": ["10.5%", "18.0%", "25.6%", "35.4%",
                   "45.0%", "48.8%", "69.2%", "72.1%"],
        "Top-5": ["26.5%", "39.0%", "43.1%", "54.4%",
                   "63.1%", "69.4%", "84.6%", "86.5%"],
        "Temps": ["79.9 ms", "11.6 ms", "183.0 ms", "19.7 ms",
                   "374.6 ms", "27.0 ms", "511.3 ms", "31.5 ms"],
    })
    st.dataframe(cluster_data, hide_index=True, use_container_width=True)

    st.image("img/strategie1_filtre_cluster.png",
             caption="Impact du pré-filtre cluster — précision et temps de réponse")

    st.markdown("---")

    # ─── Métriques clés
    st.markdown("## Métriques clés (avec filtre cluster, 6 gestes)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Top-1", value="72.1%", delta="+2.9 pp vs sans filtre")
    with col2:
        st.metric(label="Top-5", value="86.5%", delta="+1.9 pp vs sans filtre")
    with col3:
        st.metric(label="Temps moyen", value="31.5 ms", delta="-94% vs sans filtre", delta_color="inverse")

    st.markdown("---")

    # ─── Conclusion
    st.markdown("## Conclusion")
    st.markdown("""
    - **Le Trie** filtre rapidement les recettes compatibles avec la séquence observée
    - **Le scoring multi-critères** classe les candidates selon coverage, ordre et complétion
    - **Le filtre cluster** améliore à la fois la précision (+3 à +8 pp) et la vitesse (jusqu'à 16× plus rapide)

    **Limite identifiée** : à 3 gestes, la précision Top-1 reste à **18%** car de nombreuses
    recettes partagent les mêmes verbes communs (`chop, mix, heat...`). Cette ambiguïté est
    structurelle — le pipeline LLM normalise plusieurs recettes différentes vers la même
    séquence d'actions. La Méthode 2 (Random Forest) cherche à dépasser cette limite.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 : STRATÉGIE 3
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Methode 2 — Classification":
    st.markdown('<div class="main-title">Methode 2 — Classification hiérarchique</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Objectif
    Apprendre automatiquement à **prédire le cluster** d'une recette à partir d'une
    séquence partielle de gestes, puis utiliser ce cluster prédit comme **filtre intelligent**
    avant la Méthode 1.

    Cette méthode remplace l'**oracle** de la Méthode 1 
    par un système réaliste, utilisable en production.
    """)

    col_warn, _ = st.columns([2, 1])
    with col_warn:
        st.warning(
            "**Pourquoi classifier le cluster, pas la recette directement ?** "
            "34 591 recettes = 34 591 classes possibles, avec 1 seul exemple par classe. "
            "Aucun classifieur (DTC, RF) ne peut apprendre dans ces conditions. "
            "→ On apprend les **25 clusters** (au lieu des 34 591 recettes), avec ~1 400 exemples par cluster."
        )

    st.markdown("---")
    st.markdown("## Pipeline en 4 étapes")

    # ─── Étape 1 : Construction du dataset
    st.markdown("### 1. Construction du dataset d'entraînement")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        On crée des **exemples d'entraînement** en prenant des **préfixes** de longueurs variées
        (2, 3, 4, 5, 6, 7, 8 gestes) pour les 3 variantes de chaque recette.

        **Pourquoi ?** Le classifieur doit apprendre à reconnaître le cluster
        depuis **n'importe quelle longueur** d'observation, pas seulement les séquences complètes.

        - Pour `[chop, mix, heat, season, serve]` :
            - `[chop, mix]` → cluster 20
            - `[chop, mix, heat]` → cluster 20
            - `[chop, mix, heat, season]` → cluster 20
            - `[chop, mix, heat, season, serve]` → cluster 20
        """)
    with col2:
        st.info("""
        **Volume d'entraînement**

        - **34 591** recettes
        - × 3 variantes
        - × ~5 longueurs de préfixe

        = **549 088 exemples**

        ~16 exemples / recette
        """)

    st.markdown("---")

    # ─── Étape 2 : DTC
    st.markdown("### 2. Decision Tree Classifier (DTC)")

    st.markdown("""
    On entraîne un **arbre de décision** pour prédire le cluster (25 classes possibles)
    à partir d'une séquence vectorisée par TF-IDF.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Méthodologie rigoureuse :**

        1. **Split stratifié 80/20** (train/test) par cluster
        2. **Sous-échantillon 10K** pour Grid Search rapide
        3. **Grid Search** sur 4 hyperparamètres × 5-fold CV = **360 entraînements**
        4. **Entraînement final** sur tout le train (440 K exemples)
        5. **K-Fold CV (5 folds)** pour valider la robustesse

        **Hyperparamètres testés** : `max_depth`, `min_samples_split`,
        `min_samples_leaf`, `criterion`
        """)
    with col2:
        st.success("""
        **Résultats DTC**

        - Accuracy test : **52.4%**
        - K-Fold CV : 52.0% ± 0.24%
        - Train accuracy : 54.9%
        - **Gap train-test : 2.5 pp**

        → Pas d'overfitting
        """)

    st.image("img/strategie3_dtc.png",
             caption="DTC — Matrice de confusion + Top 20 verbes importants")

    st.markdown("""
    **Interprétation de la feature importance** : les verbes techniques comme `knead`,
    `boil`, `beat`, `roll`, `brown`, `mash` dominent — le DTC apprend les **bonnes
    intuitions culinaires**.
    """)

    st.markdown("---")

    # ─── Étape 3 : Random Forest
    st.markdown("### 3. Random Forest (RF)")

    st.markdown("""
    Random Forest = **100 arbres de décision** entraînés sur des sous-échantillons
    différents, qui votent collectivement. Plus robuste qu'un DTC seul.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Mêmes étapes méthodologiques que le DTC** :

        1. Sous-échantillon 10K pour Grid Search
        2. **Grid Search** sur 4 hyperparamètres × 3-fold CV = 108 entraînements
        3. Entraînement final sur 440 K exemples
        4. K-Fold CV (5) pour validation

        **Hyperparamètres testés** : `n_estimators`, `max_depth`,
        `max_features`, `min_samples_split`

        **Optimaux trouvés** : `n_estimators=100`, `max_depth=None`,
        `max_features='sqrt'`, `min_samples_split=5`
        """)
    with col2:
        st.success("""
        **Résultats RF**

        - Accuracy test : **57.6%**
        - K-Fold CV : 56.8% ± 0.18%
        - Temps entraînement : 105.5s
        - **+5.2 pp** vs DTC

        → Bagging compense l'overfitting
        """)

    st.image("img/strategie3_rf.png",
             caption="RF — Matrice de confusion + Top 20 verbes importants")

    st.markdown("---")

    # ─── Comparaison DTC vs RF
    st.markdown("### Comparaison DTC vs RF")

    comp_data = pd.DataFrame({
        "Métrique": ["Accuracy (test set)", "CV mean", "CV std", "Temps entraînement", "Grid Search time"],
        "DTC": ["52.4%", "52.0%", "0.24%", "7.5 s", "9.2 s"],
        "Random Forest": ["**57.6%**", "**56.8%**", "0.18%", "105.5 s", "26.1 s"],
        "Delta": ["+5.2 pp", "+4.8 pp", "—", "14× plus lent", "3× plus lent"],
    })
    st.dataframe(comp_data, hide_index=True, use_container_width=True)

    st.image("img/strategie3_comparaison.png",
             caption="Comparaison DTC vs Random Forest")

    st.info(
        "**Conclusion étape 3** : RF gagne nettement en précision (+5 pp) au prix d'un "
        "entraînement 14× plus long. Le compromis est largement favorable au RF "
        "pour notre cas d'usage."
    )

    st.markdown("---")

    # ─── Étape 4 : Intégration avec le Trie
    st.markdown("### 4. Intégration avec le Trie (pipeline complet)")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        Le RF prédit le cluster, qui devient le **filtre** de la Méthode 1 :

        ```
        Séquence observée → RF → cluster prédit
                                       ↓
                          Filtre → Trie → Scoring → Top-5
        ```

        Ce pipeline est **l'équivalent réaliste de l'oracle** utilisé en Méthode 1
        (qui trichait avec la vraie réponse).
        """)
    with col2:
        st.success("""
        **Précision du RF à prédire le cluster**

        - 3 gestes : 57.0%
        - 4 gestes : 83.6%
        - 5 gestes : 97.5%
        - **6 gestes : 99.0%** 

        → Quasi-parfait à 6 gestes
        """)

    st.markdown("---")

    # ─── Comparaison finale 3 systèmes
    st.markdown("## Comparaison finale — 3 systèmes")

    st.markdown("""
    On compare 3 versions du pipeline complet sur **200 recettes** :
    - **A — Sans filtre** : Trie + scoring uniquement (Méthode 1 sans cluster)
    - **B — Oracle** : on utilise le vrai cluster de la recette comme filtre (Méthode 1 complète) 
    - **C — RF prédit (réaliste)** : pipeline réel utilisable en production
    """)

    final_data = pd.DataFrame({
        "Gestes": [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
        "Système": [
            "Sans filtre", "Oracle", "RF prédit",
            "Sans filtre", "Oracle", "RF prédit",
            "Sans filtre", "Oracle", "RF prédit",
            "Sans filtre", "Oracle", "RF prédit",
        ],
        "Top-1": [
            "10.5%", "18.5%", "15.0%",
            "25.1%", "34.9%", "31.8%",
            "46.2%", "50.6%", "49.4%",
            "69.2%", "73.1%", "**72.1%**",
        ],
        "Top-5": [
            "27.5%", "40.5%", "32.0%",
            "45.1%", "54.4%", "48.7%",
            "61.9%", "70.6%", "68.8%",
            "82.7%", "85.6%", "**84.6%**",
        ],
        "Temps": [
            "78.8 ms", "10.8 ms", "28.3 ms",
            "181.7 ms", "18.9 ms", "33.8 ms",
            "377.4 ms", "26.5 ms", "40.3 ms",
            "513.8 ms", "32.0 ms", "**44.4 ms**",
        ],
    })
    st.dataframe(final_data, hide_index=True, use_container_width=True)

    st.image("img/strategie3_final.png",
             caption="Comparaison finale — Sans filtre / Oracle / RF prédit (3 à 6 gestes)")

    st.markdown("---")

    # ─── Métriques clés
    st.markdown("## Métriques clés (RF prédit, 6 gestes)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Top-1", value="72.1%", delta="=oracle")
    with col2:
        st.metric(label="Top-5", value="84.6%", delta="-1.0 pp vs oracle")
    with col3:
        st.metric(label="Cluster correct", value="99.0%", delta="à 6 gestes")
    with col4:
        st.metric(label="Temps", value="44.4 ms", delta="≈ Oracle (32 ms)")

    st.markdown("---")

    # ─── Conclusion
    st.markdown("## Conclusion")
    st.markdown("""
    - **Random Forest** est nettement meilleur que **DTC** (+5 pp) pour la prédiction du cluster
    - À **6 gestes observés**, le RF prédit le bon cluster **99% du temps** → quasi-oracle
    - Le pipeline complet **(RF → cluster → Trie → scoring)** atteint **72.1% Top-1** et **84.6% Top-5**
    en seulement **44 ms par requête**
    - **Écart RF vs Oracle : seulement 1 pp** → le système réaliste est **indistinguable** d'un système
    qui tricherait avec la vraie réponse

    **La Méthode 2 valide l'approche hiérarchique** : on n'a pas besoin de connaître
    la vraie recette à l'avance — il suffit de bien classifier le cluster, ce que le RF fait excellemment.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 : COMPARAISON FINALE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Comparaison finale":
    st.markdown('<div class="main-title">Comparaison finale & Synthèse</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Vue d'ensemble du projet
    On a construit un système de **reconnaissance de recettes** à partir d'une
    **séquence partielle de gestes** observés chez un cuisinier. Trois briques principales :
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Clustering")
        st.markdown("""
        - **TF-IDF + PCA + K-Means**
        - 25 clusters obtenus
        - Familles culinaires émergentes
        """)
    with col2:
        st.markdown("#### Méthode 1")
        st.markdown("""
        - **Trie + Scoring**
        - Filtre par cluster oracle
        - 72.1% Top-1 à 6 gestes
        """)
    with col3:
        st.markdown("#### Méthode 2")
        st.markdown("""
        - **DTC + Random Forest**
        - Prédit le cluster (57% → 99%)
        - Remplace l'oracle
        """)

    st.markdown("---")

    # ─── Tableau récapitulatif des 3 systèmes
    st.markdown("## Tableau récapitulatif — 3 systèmes (200 recettes, 6 gestes)")

    recap_data = pd.DataFrame({
        "Système": ["Sans filtre", "Oracle (triche)", "RF prédit (réaliste)"],
        "Description": [
            "Trie + scoring sur tout le dataset",
            "Filtre cluster avec la vraie réponse",
            "Filtre cluster prédit par Random Forest",
        ],
        "Top-1": ["69.2%", "73.1%", "**72.1%**"],
        "Top-5": ["82.7%", "85.6%", "**84.6%**"],
        "Temps": ["513.8 ms", "32.0 ms", "**44.4 ms**"],
        "Production ?": ["Lent", "Impossible", "**Oui**"],
    })
    st.dataframe(recap_data, hide_index=True, use_container_width=True)

    st.success(
        "**Résultat clé** : le système RF prédit (réaliste) atteint **99% des performances "
        "de l'oracle théorique**, en seulement 44 ms par requête. Le pipeline est utilisable "
        "en production."
    )

    st.markdown("---")

    # ─── Évolution de la précision selon le nombre de gestes
    st.markdown("## Évolution de la précision selon le nombre de gestes observés")

    evol_data = pd.DataFrame({
        "Gestes": [3, 4, 5, 6],
        "Sans filtre Top-1": ["10.5%", "25.1%", "46.2%", "69.2%"],
        "Oracle Top-1": ["18.5%", "34.9%", "50.6%", "73.1%"],
        "RF prédit Top-1": ["15.0%", "31.8%", "49.4%", "**72.1%**"],
        "RF cluster correct": ["57.0%", "83.6%", "97.5%", "**99.0%**"],
    })
    st.dataframe(evol_data, hide_index=True, use_container_width=True)

    st.image("img/strategie3_final.png",
             caption="Comparaison finale — précision et précision du RF à prédire le cluster")

    st.info(
        "**Lecture** : à 6 gestes, le RF prédit le bon cluster 99% du temps. "
        "Le système réaliste (RF prédit) est à seulement **1 point de l'oracle théorique**."
    )

    st.markdown("---")

    # ─── Métriques clés finales
    st.markdown("## Métriques clés du système final")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Top-1 (6 gestes)", value="72.1%", delta="+62 pp vs aléatoire")
    with col2:
        st.metric(label="Top-5 (6 gestes)", value="84.6%", delta="Très utilisable")
    with col3:
        st.metric(label="Temps / requête", value="44 ms", delta="Temps réel")
    with col4:
        st.metric(label="Cluster correct", value="99.0%", delta="à 6 gestes")

    st.markdown("---")

    # ─── Limites identifiées
    st.markdown("## Limites identifiées")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Précision plafonnée à 3 gestes")
        st.markdown("""
        À **3 gestes observés**, le Top-1 reste à **15-18%**.

        **Raison** : le pipeline LLM normalise plusieurs recettes différentes
        vers la **même séquence d'actions** (ex : `[chop, mix, heat]`).

        Cette ambiguïté est **structurelle** — aucun scoring ou filtre cluster
        ne peut distinguer 2 recettes qui ont **exactement les mêmes 3 verbes**.
        """)
    with col2:
        st.markdown("### Doublons sémantiques dans le dataset")
        st.markdown("""
        Plusieurs centaines de recettes peuvent partager les **3-5 mêmes gestes**.

        Exemples observés :
        - 5 salades différentes → `[chop, mix, heat, season, serve]`
        - 4 ragoûts similaires → `[chop, mix, simmer, serve]`

        Le système renvoie alors **des recettes équivalentes**, ce qui est correct
        mais pas toujours la "bonne" recette.
        """)

    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Navigation entre pages
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
_idx = PAGES.index(page)
nav_left, nav_mid, nav_right = st.columns([1, 4, 1])
with nav_left:
    if _idx > 0:
        st.button(f"← {PAGES[_idx - 1]}", on_click=go_prev, use_container_width=True)
with nav_mid:
    st.markdown(
        f"<p style='text-align: center; color: #95A5A6; padding-top: 0.5rem;'>"
        f"Page {_idx + 1} / {len(PAGES)}</p>",
        unsafe_allow_html=True
    )
with nav_right:
    if _idx < len(PAGES) - 1:
        st.button(f"{PAGES[_idx + 1]} →", on_click=go_next, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer fixe en bas d'ecran
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        color: #95A5A6;
        font-size: 0.8rem;
        padding: 8px 0;
        background-color: rgba(14, 17, 23, 0.95);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 999;
    }
    /* Padding bas du contenu principal pour ne pas etre cache par le footer */
    .main .block-container {
        padding-bottom: 60px;
    }
</style>
<div class="fixed-footer">Stage Projet I - david NIDMINA - 2026</div>
""", unsafe_allow_html=True)
