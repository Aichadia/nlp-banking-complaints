# 🧠 Analyse NLP des Plaintes Bancaires — Trustpilot × 8 banques françaises

> Pipeline NLP complet sur 34 026 avis clients : scraping, preprocessing, topic modeling, classification thématique Zero-Shot et analyse de sentiment — pour identifier automatiquement les drivers d'insatisfaction dans le secteur bancaire.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=flat-square&logo=selenium&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6B23?style=flat-square)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat-square)

---

## 📌 Contexte & problème business

Les banques reçoivent des milliers d'avis en ligne chaque mois — mais les analyser manuellement est impossible à l'échelle. Ce projet répond à la question : **comment identifier automatiquement les sources d'insatisfaction client dans le secteur bancaire à partir de données non structurées ?**

Mémoire de Master 2 *Économie Appliquée — MASERATI* (UPEC), réalisé en binôme. Les données ont été collectées via **web scraping** sur Trustpilot, couvrant 8 banques françaises (traditionnelles et en ligne) sur **9 ans** (2016–2025).

---

## 📊 Résultats

### Performance des modèles de sentiment

| Modèle | Vectorisation | Accuracy | F1 macro | AUC |
|--------|--------------|----------|----------|-----|
| **SVM-RBF** | CamemBERT embeddings | **95.6%** | **95.2%** | **0.99** |
| XGBoost | TF-IDF | 92.9% | 92.0% | 0.976 |

> Le modèle CamemBERT + SVM-RBF a été optimisé via **Optuna** avec cross-validation stratifiée. XGBoost + TF-IDF permet le passage à l'échelle sur les 32 000+ avis avec des performances très proches.

### Insights business identifiés

| Type de banque | % avis 1 étoile | % avis 5 étoiles | Drivers d'insatisfaction |
|---------------|----------------|-----------------|--------------------------|
| **Banques traditionnelles** | 66–77% | 15–27% | Délais administratifs, frais bancaires, relation conseiller |
| **Banques en ligne** | 8–53% | 65–81% | Services numériques, cartes bancaires, service client à distance |

> 💡 **Insight clé** : La Société Générale concentre 77% d'avis 1 étoile. Boursobank affiche 81% d'avis 5 étoiles. Les thèmes les plus corrélés aux notes négatives (1–2 étoiles) sont **les délais de traitement, les problèmes de carte bancaire et les litiges** — identifiés automatiquement via classification Zero-Shot (mDeBERTa).

---

## 🔍 Ce que ce projet démontre

- **Collecte autonome** : scraping de 34 026 avis sur Trustpilot (Selenium + ChromeDriverManager), sans API officielle
- **Pipeline NLP complet** : tokenisation, lemmatisation (spaCy `fr_core_news_lg`), suppression stopwords enrichie, N-grams (Gensim), vectorisation TF-IDF et CamemBERT
- **Approche hybride** : LDA exploratoire → annotation semi-automatisée via OpenAI API → classification Zero-Shot large scale (mDeBERTa, HuggingFace) sur 15 catégories thématiques
- **Comparaison de stratégies** : TF-IDF vs embeddings contextuels, trade-off précision/scalabilité sur 32 000+ documents
- **Lecture métier** : croisement catégories prédites × notes pour identifier les leviers d'action prioritaires par type de banque

---

## 🛠️ Stack technique

| Catégorie | Outils |
|-----------|--------|
| Collecte | Python, Selenium, ChromeDriverManager |
| Preprocessing | spaCy (`fr_core_news_lg`), Gensim, NLTK |
| Vectorisation | TF-IDF (Scikit-learn), CamemBERT (HuggingFace Transformers) |
| Topic Modeling | LDA (Gensim), pyLDAvis |
| Classification | mDeBERTa-v3 (Zero-Shot), OpenAI API |
| Sentiment | SVM-RBF, XGBoost, Random Forest, Logistic Regression, Naive Bayes |
| Optimisation | Optuna, stratified cross-validation |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Environnement | Google Colab |

---

## 📁 Structure du repo

```
📦 nlp-banking-complaints
 ┣ 📂 data/
 ┃ ┣ raw/                                      # Avis bruts scrapés (un fichier par banque)
 ┃ ┗ processed/                                # Corpus nettoyés et lemmatisés
 ┣ 📂 notebooks/
 ┃ ┣ 01_scraping_trustpilot.ipynb              # Collecte Trustpilot via Selenium
 ┃ ┣ 02_preprocessing.ipynb                    # Nettoyage, lemmatisation spaCy, N-grams Gensim
 ┃ ┣ 03_eda_exploratoire.ipynb                 # Stats descriptives, word clouds, distribution notes
 ┃ ┣ 04_vectorisation_tfidf.ipynb              # Vectorisation TF-IDF (matrice creuse)
 ┃ ┣ 05_lda_banques_en_ligne.ipynb             # Topic Modeling LDA — banques en ligne
 ┃ ┣ 06_lda_banques_traditionnelles.ipynb      # Topic Modeling LDA — banques traditionnelles
 ┃ ┣ 07_annotation_openai_gpt.ipynb            # Annotation semi-automatique via OpenAI API
 ┃ ┣ 08_zero_shot_mdeberta.ipynb               # Classification Zero-Shot (mDeBERTa, HuggingFace)
 ┃ ┣ 09_sentiment_tfidf_xgboost.ipynb          # Analyse de sentiment TF-IDF + XGBoost
 ┃ ┗ 10_sentiment_camembert_svm.ipynb          # Analyse de sentiment CamemBERT + SVM-RBF
 ┣ 📂 outputs/
 ┃ ┣ wordclouds/                               # Word clouds banques en ligne vs traditionnelles
 ┃ ┣ confusion_matrices/                       # Matrices de confusion SVM-RBF et XGBoost
 ┃ ┗ topic_distributions/                      # Heatmaps catégories × notes
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

---

## 🚀 Comment reproduire

```bash
# 1. Cloner le repo
git clone https://github.com/Aichadia/nlp-banking-complaints.git
cd nlp-banking-complaints

# 2. Installer les dépendances
pip install selenium webdriver-manager spacy gensim scikit-learn xgboost \
            transformers torch optuna matplotlib seaborn wordcloud openai

# 3. Télécharger le modèle spaCy
python -m spacy download fr_core_news_lg

# 4. Lancer les notebooks dans l'ordre
# (01 → scraping, 02 → preprocessing, etc.)
```

> ⚠️ Le scraping est sensible aux changements de structure HTML de Trustpilot. Les données brutes sont disponibles dans `/data/raw/`.

---

## 📈 Démarche

1. **Scraping** — Collecte automatisée de 34 026 avis sur 8 banques via Selenium ; gestion des pages dynamiques JavaScript et des erreurs de chargement
2. **Preprocessing** — Nettoyage, lemmatisation spaCy, suppression stopwords enrichie (prénoms INSEE, termes génériques), réduction vocabulaire de 81 194 → 18 721 termes
3. **EDA** — Distribution des notes par banque, analyse des longueurs de commentaires, word clouds et bi/trigrammes comparatifs (banques en ligne vs traditionnelles)
4. **Topic Modeling** — LDA exploratoire (Gensim + pyLDAvis), puis annotation de 100 avis via OpenAI API, puis Zero-Shot Classification (mDeBERTa) sur 15 catégories thématiques
5. **Analyse de sentiment** — Comparaison TF-IDF vs CamemBERT, sélection SVM-RBF (CamemBERT) et XGBoost (TF-IDF) après tuning Optuna
6. **Analyse métier** — Croisement catégories prédites × notes pour identifier les drivers d'insatisfaction par type de banque

---

## 👩‍💻 Auteure

**Aicha DIAKITE** — en collaboration avec Koffi AMEMOU
🎓 Master 2 Économie Appliquée — MASERATI — UPEC (2024-2025)
🔗 [LinkedIn](https://linkedin.com/in/aicha-diakite) · [GitHub](https://github.com/Aichadia)
