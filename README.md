
````markdown
# Heart Disease Prediction System

This is a **Heart Disease Prediction System** built using machine learning models. The system predicts whether a patient is likely to have heart disease based on clinical and diagnostic features. The application includes a **Streamlit frontend** for easy user interaction and stores all prediction records for future reference.

---

## 📊 Dataset

The dataset used for this project is taken from Kaggle:  
[Heart Disease Dataset - Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2/data)

The dataset contains features such as age, sex, blood pressure, cholesterol levels, ECG results, exercise-induced angina, ST depression, number of vessels, chest pain type, Thallium scan results, and slope of ST segment.

---

## 📝 Notebook

The full **exploratory data analysis, feature engineering, model training, and evaluation** is available here:  
[Heart Disease Predictor Notebook](https://www.kaggle.com/code/aroshabakhtawar/heart-disease-predictor)

---

## 🤖 Models Experimented

Several machine learning models were trained and evaluated:

| Model                 | CV AUC   | CV Std   | Train AUC | Val AUC | Time (s) |
|-----------------------|----------|----------|-----------|---------|----------|
| CatBoost              | 0.955063 | 0.000437 | 0.959798  | 0.955504| 379.9    |
| LightGBM              | 0.954714 | 0.000415 | 0.955644  | 0.955062| 34.3     |
| XGBoost               | 0.954640 | 0.000407 | 0.958845  | 0.955029| 19.6     |
| Gradient Boosting      | 0.953769 | 0.000431 | 0.953930  | 0.954000| 399.9    |
| AdaBoost               | 0.952817 | 0.000486 | 0.952775  | 0.952965| 91.0     |
| Logistic Regression    | 0.952679 | 0.000400 | 0.952590  | 0.952988| 6.6      |
| LDA                    | 0.951338 | 0.000358 | 0.951266  | 0.951509| 5.0      |
| Random Forest          | 0.948888 | 0.000503 | 1.000000  | 0.949112| 487.5    |
| Extra Trees            | 0.947843 | 0.000388 | 1.000000  | 0.947920| 399.0    |
| Decision Tree          | 0.824237 | 0.000568 | 1.000000  | 0.823487| 17.6     |
| Naive Bayes            | 0.795056 | 0.001122 | 0.794722  | 0.796364| 2.0      |

**CatBoost** was selected as the final model due to its superior performance in cross-validation AUC and stability.

---

## ⚙️ Features Used

- `id`
- `Age`
- `Sex`
- `BP` (Blood Pressure)
- `Cholesterol`
- `FBS over 120` (Fasting Blood Sugar)
- `Max HR` (Maximum Heart Rate)
- `Exercise Angina`
- `ST Depression`
- `Number of vessels fluro`
- `Chest Pain Type` (one-hot encoded)
- `EKG Results` (one-hot encoded)
- `Thallium` (one-hot encoded)
- `Slope of ST` (one-hot encoded)

---

## 🛠️ Installation & Usage

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
````

2. **Install required packages**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

4. **Open your browser** at the URL provided by Streamlit (usually `http://localhost:8501`).

---

## 💻 Functionality

* User inputs patient details through the Streamlit web interface.
* The app processes the inputs, applies one-hot encoding for categorical features, and predicts **heart disease probability** using the trained **CatBoost model**.
* Predictions and probabilities are displayed in real-time.
* All input data and predictions are **saved in a CSV file** for record keeping.
* The frontend highlights important features contributing to the prediction (optional with SHAP).

---

## 🧪 Model Training

The following models were experimented with:

* Logistic Regression
* LDA (Linear Discriminant Analysis)
* Naive Bayes
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* AdaBoost
* XGBoost
* LightGBM
* CatBoost

Hyperparameters were tuned to maximize **cross-validation AUC**. CatBoost achieved the best results.

---

## 📂 Files in the Repository

* `app.py` – Streamlit app for real-time heart disease prediction.
* `CatBoostModel.cbm` – Trained CatBoost model.
* `heart_disease_records.csv` – Saved predictions.
* `requirements.txt` – Python dependencies.
* `notebook.ipynb` – Model training and evaluation notebook.

---

## 🔗 References

* Dataset: [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2/data)
* Notebook: [Heart Disease Predictor](https://www.kaggle.com/code/aroshabakhtawar/heart-disease-predictor)

---

## 📌 Author

**Arosha Bakhtawar**

* Kaggle: [@aroshabakhtawar](https://www.kaggle.com/aroshabakhtawar)
* GitHub: [https://github.com/aroshabakhtawar](https://github.com/aroshabakhtawar)

---



