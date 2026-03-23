# 🤖 Machine Learning Lab Environment (KNN, Regression & Neural Networks)

A structured and practical machine learning project developed from Google Colab notebooks and organized into a clean, professional local environment.

This repository covers core **Machine Learning algorithms and workflows**, including:

* K-Nearest Neighbors (KNN)
* Linear Regression
* Logistic Regression
* Neural Networks (TensorFlow/Keras)
* Model Evaluation & Fine-Tuning

It provides a **step-by-step learning pipeline**, from data preprocessing to advanced modeling.

---

# 🚀 Quick Start

## 🔧 1. Install Dependencies

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 📓 2. Launch Jupyter Notebook

```powershell
.venv\Scripts\jupyter notebook
```

Or:

```powershell
.venv\Scripts\jupyter lab
```

---

## ▶️ 3. Run Notebooks

Open files in the `notebooks/` folder and execute cells using:

```
Shift + Enter
```

---

# 📁 Project Structure

This project was originally built in Google Colab and has been reorganized for clarity and professional use.

```
├── 📓 notebooks/                # Main learning notebooks (ordered)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_linear_regression.ipynb
│   ├── 03_knn_classifier.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_neural_networks.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_finetuning.ipynb
│   ├── 08_advanced_tasks.ipynb
│   └── FINAL_TASK.ipynb
│
├── 📊 data/                     # Datasets
├── 📁 outputs/                  # Results, plots, predictions
├── 📚 docs/                     # Extra notes/documentation (optional)
├── 🐍 .venv/                    # Virtual environment
├── 📄 requirements.txt          # Dependencies
└── 📄 README.md
```

---

# 🧠 Learning Flow

Follow the notebooks in order for the best understanding:

1. **Data Preprocessing**
2. **Linear Regression**
3. **KNN Classifier**
4. **Logistic Regression**
5. **Neural Networks**
6. **Model Evaluation**
7. **Fine-Tuning**
8. **Advanced Tasks**
9. **Final Integrated Project**

Each notebook builds on the previous one.

---

# 📚 Tasks Covered

---

## 🔹 Task 1: Data Preprocessing

* Handle missing values (mean, median, drop)
* Encode categorical variables (One-hot, Label Encoding)
* Normalize / Standardize features
* Train-test split

---

## 🔹 Task 2: Linear Regression

* Train regression model
* Interpret coefficients
* Evaluate using:

  * Mean Squared Error (MSE)
  * R² Score

---

## 🔹 Task 3: KNN Classifier

* Train KNN model
* Experiment with different **K values**
* Evaluate using:

  * Accuracy
  * Confusion Matrix
  * Precision & Recall

---

## 🔹 Task 4: Logistic Regression

* Binary classification
* Interpret coefficients & odds ratio
* Evaluate using:

  * Accuracy
  * Precision & Recall
  * ROC Curve

---

## 🔹 Task 5: Neural Networks

* Build deep learning models (Keras)
* Train using backpropagation
* Evaluate accuracy
* Visualize loss curves

---

## 🔹 Task 6–8: Advanced Topics

* Model evaluation techniques
* Fine-tuning models
* Experimentation and improvements

---

## 🔹 Final Task

* Combines all concepts into a complete ML pipeline project

---

# ⚙️ Technologies Used

* **Python 3.x**
* **pandas** – Data processing
* **NumPy** – Numerical computation
* **scikit-learn** – ML algorithms
* **TensorFlow / Keras** – Deep learning
* **matplotlib** – Visualization

---

# 📊 Example: Model Evaluation

```python
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

# 🧠 Key Learning Outcomes

By completing this project, you will understand:

* Data preprocessing techniques
* Supervised learning models
* Model evaluation metrics
* Differences between:

  * KNN (distance-based)
  * Regression (linear models)
  * Neural networks (non-linear learning)

---

# 🛠️ Troubleshooting

### ❌ Module not found

👉 Activate virtual environment and reinstall dependencies

### ❌ Jupyter not opening

```powershell
.venv\Scripts\jupyter notebook
```

### ❌ Wrong kernel selected

👉 Choose:

```
Python (.venv)
```

---

# 🎯 How to Use This Repo

* Follow notebooks in order
* Run all cells step-by-step
* Modify parameters (K, epochs, etc.)
* Test with your own datasets

---

# 🎓 Disclaimer

This project is for **educational purposes only** and demonstrates core machine learning concepts.

---

# 🎉 You're Ready!

* Explore the notebooks
* Experiment with models
* Build your own ML projects 🚀

---

**Happy Learning & Building! 🤖🔥**
