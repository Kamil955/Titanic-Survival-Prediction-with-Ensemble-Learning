# 🚢 Titanic Survival Prediction with Ensemble Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB2F2F?logo=xgboost&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Accuracy](https://img.shields.io/badge/Accuracy-81%25-brightgreen)

## 📄 Project Overview
This project aims to predict the survival of passengers on the RMS Titanic using advanced machine learning techniques. By analyzing demographic and ticket data, I developed a predictive model that achieves an accuracy of **~80%** on the test dataset.

The core of the solution relies on **Feature Engineering** and **Ensemble Learning**, combining the strengths of Random Forest, XGBoost, and Logistic Regression via a Voting Classifier.

## 📊 Methodology

### 1. Data Preprocessing & Cleaning
- **Missing Values:** Imputed `Age` using median and delated two `Embarked` empty rows.
- **Categorical Encoding:** Applied One-Hot Encoding to `Sex`, `Embarked`, and `Deck`.
- **Scaling:** Used `StandardScaler` for linear models.

### 2. Feature Engineering
Key features created to improve model performance:
- **`Title`**: Extracted from names (Mr, Mrs, Master, etc.) to capture social status.
- **`FamilySize`**: Combined `SibSp` + `Parch` + 1.
- **`FarePerPerson`**: Normalized ticket price based on family size.
- **`IsAlone`**: Binary flag for passengers traveling solo.

### 3. Models Used
I trained and tuned three distinct models:
1.  **Random Forest Classifier** (Optimized for `min_samples_leaf` and `n_estimators`)
2.  **XGBoost Classifier** (Gradient Boosting for capturing complex patterns)
3.  **Logistic Regression** (Baseline linear model)

### 4. Ensemble Strategy: Soft Voting (The Solution 🏆)
Instead of a simple majority vote (Hard Voting), I implemented a **Soft Voting** mechanism.
This approach averages the **predicted probabilities** of the two strongest models (Random Forest + XGBoost).

$$P(Survival) = \frac{P_{RF}(Survival) + P_{XGB}(Survival)}{2}$$

If the averaged probability is $> 0.5$, the passenger is predicted to survive. This method proved superior as it accounts for the *confidence* of each model, not just the binary decision.
## 📈 Results

| Model | Accuracy (Test) | Notes |
|-------|-----------------|-------|
| Logistic Regression | 76% | Baseline |
| Random Forest | 79% | After Hyperparameter Tuning |
| XGBoost | 79% | Strong standalone performance |
| **Voting Ensemble** | **81%** | **Final Submission Model** |

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Libraries:**
    - `pandas`, `numpy` (Data Manipulation)
    - `matplotlib`, `seaborn` (Visualization)
    - `scikit-learn` (Modeling, GridSearch)
    - `xgboost` (Gradient Boosting)
    - `joblib` (Model Persistence)
