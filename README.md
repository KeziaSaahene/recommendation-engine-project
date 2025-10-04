# Recommendation System Project

This project focuses on designing and evaluating a **personalized recommendation system** using historical user interaction data. The **CRISP-DM methodology** guides the process from business understanding to deployment.

---

## 🛠️ Tools & Technologies

- **Python** – core programming language  
- **Pandas, NumPy, SciPy** – data manipulation & matrix factorization  
- **Scikit-learn** – preprocessing, evaluation metrics, anomaly detection  
- **TensorFlow / Keras** – deep learning (CNN & Autoencoder models)  
- **Matplotlib & Seaborn** – data visualization  
- **Implicit (ALS)** – collaborative filtering (if installed)  
- **GitHub** – version control & progress tracking  

---

## 📁 Dataset

The dataset contains three main files:

- `events.csv` → user interactions (timestamp, visitorid, event, itemid, transactionid)  
- `item_properties_combined.csv` → item metadata (timestamp, itemid, categoryid, value)  
- `category_tree.csv` → hierarchical item relationships (child, parent)  

---

## 📊 CRISP-DM Framework

### 1. Business Understanding

**Objective:**

- Build a personalized recommendation system.  
- Detect abnormal users (bots, fraud, unusual patterns).  

**Business Questions & Insights:**

| Question | Insight |
|----------|---------|
| How many unique visitors are interacting with the platform? | 1,407,580 |
| On average, how many events does a visitor generate? | 1.96 |
| How many unique items does a visitor typically interact with? | 1.52 |
| Peak hours of visitor activity? | 7 PM |
| Peak days of the week? | Tuesday |
| Total parent categories? | 362 |
| Top parent categories with most children? | 250 |
| Total number of categories? | 1,092 |
| Total unique items? | 5,369k |
| Average unique items per category? | 4,917.18 |
| Top 10 most interacted-with categories? | See analysis |

---

### 2. Data Understanding

- **Events dataset:** 8.5M+ rows of user interactions  
- **Item properties:** metadata requiring latest-value extraction  
- **Category tree:** hierarchical grouping of items  

**Key Insights:**

- Events are heavily skewed toward views  
- Transactions form a small but valuable subset  
- Metadata preprocessing was crucial for labeling  

---

### 3. Data Preparation

- **Data Loading:** Imported datasets (`events.csv`, `category_tree.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`)  
- **Merging:** Combined `item_properties_part1.csv` and `item_properties_part2.csv` into a single consolidated file  
- **Exploration & Cleaning:**  
  - Checked dataset structure (shape, data types, missing values, duplicates)  
  - Validated unique identifiers (`visitorid`, `itemid`, `transactionid`)  
- **Feature Engineering:** Converted timestamps from milliseconds to datetime format  
- **Dataset Summarization:** Generated descriptive statistics and unique counts for all datasets  

---

### 4. Modeling

**CNN Model:**

- Built a tuned CNN model on user-item interaction sequences using tokenized item properties  
- Training used **early stopping** and **Adam optimizer**  

**Anomaly Detection:**

- Built user-level behavioral features and trained a **CNN Autoencoder** for unsupervised anomaly detection  
- Training used **reconstruction loss (MSE)** with early stopping  

---

### 5. Evaluation

**CNN Model:**

- Metrics: Accuracy, Precision, Recall, F1-score, Recall@K, Hit Rate@K, NDCG@K  

**Anomaly Detection:**

- Flagged abnormal users based on **high reconstruction error** (top 2% threshold)  
- Error distributions visualized to highlight anomalies  

---

### 6. Deployment / Deliverables

- **Recommender System:** Saved trained CNN model, tokenizer, and label encoder as reusable artifacts  
- **Anomaly Detection:** Saved CNN Autoencoder model and feature scaler for consistent scoring  
- These artifacts enable future inference without retraining  

---

## 📌 Notes

- Ensure all dependencies in `requirements.txt` are installed  
- Some collaborative filtering features require `implicit` library  

