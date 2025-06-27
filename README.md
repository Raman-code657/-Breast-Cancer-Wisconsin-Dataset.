 Classification with Logistic Regression

## 📌 Objective
Build a **binary classifier** using **Logistic Regression** to predict whether a tumor is malignant or benign using the **Breast Cancer Wisconsin Dataset**.

---

## 📂 Dataset
- **Source**: [Breast Cancer Wisconsin Dataset - UCI / scikit-learn](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- Features include various measurements of cell nuclei from breast mass images.
- Target: `0 = Malignant`, `1 = Benign`

---

## 🔧 Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 🧪 Workflow

### 1. Data Loading
Used `load_breast_cancer()` from scikit-learn to fetch and convert the dataset into a DataFrame.

### 2. Preprocessing
- Train-test split (80-20)
- Feature standardization using `StandardScaler`

### 3. Model Training
- Applied **Logistic Regression** using `sklearn.linear_model.LogisticRegression`.

### 4. Evaluation Metrics
- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **ROC Curve & AUC Score**
- Tried custom classification thresholds (e.g., 0.6)

---

## 📊 Results

| Metric        | Value |
|---------------|--------|
| Accuracy      | 96%+   |
| Precision     | High   |
| Recall        | High   |
| ROC AUC Score | ~0.99  |


---

\
