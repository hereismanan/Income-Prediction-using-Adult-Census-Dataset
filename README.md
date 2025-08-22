# Income-Prediction-using-Adult-Census-Dataset
This project uses the **Adult Census Income dataset** from the UCI Machine Learning Repository to predict whether a person earns more than **\$50K per year**.

---

## 📌 Project Overview
- **Dataset Size**: ~32,000 records
- **Features**: Mix of categorical (workclass, education, occupation, etc.) and numerical (age, hours-per-week, capital-gain, etc.)
- **Target**: Binary classification — \`<=50K\` or \`>50K\`
- **Challenges**:
  - Missing values in categorical features
  - Mixed data types (categorical + numeric)
  - Slight class imbalance (~76% vs 24%)

---

## ⚙️ Workflow
1. **Data Cleaning**  
   - Handled missing values ("?" replaced with "Unknown")  
   - Dropped irrelevant \`fnlwgt\` column  

2. **Feature Engineering**  
   - One-hot encoding for categorical variables  
   - Numeric features passed directly (no scaling needed for trees)  

3. **Train/Test Split**  
   - 80/20 split with stratification to preserve class ratio  

4. **Models Trained**  
   - Decision Tree (baseline, interpretable)  
   - Random Forest (ensemble, robust)  
   - XGBoost (boosted trees, high performance)  

5. **Evaluation Metrics**  
   - Accuracy  
   - (Potential extensions: Precision, Recall, ROC-AUC, Fairness analysis)

---

## 📊 Results

| Model            | Accuracy |
|------------------|----------|
| Decision Tree    | **0.8265** |
| Random Forest    | **0.8481** |
| XGBoost          | **0.8776** |

---

## 💾 Deployment
- Trained pipelines (preprocessing + model) are saved using **joblib**:
  - \`decision_tree_pipeline.pkl\`
  - \`random_forest_pipeline.pkl\`
  - \`xgboost_pipeline.pkl\`

- Supports direct **user input prediction**:
  - Example: Input a single row of age, education, occupation, etc.  
  - Model outputs predicted income (\`<=50K\` or \`>50K\`)

---

## 🚀 Future Improvements
- Add hyperparameter tuning (GridSearchCV)  
- Perform fairness analysis (e.g., across gender/race groups)  
- Deploy as REST API using Flask/FastAPI  
- Build Streamlit app for interactive predictions  

---

## 📚 References
- UCI Machine Learning Repository: [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- XGBoost Documentation
- Scikit-learn Documentation

---
EOL
