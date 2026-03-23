# 🤖 Machine Learning Lab Environment (KNN, Regression & Neural Networks)

A complete local setup for learning and implementing core **Machine Learning algorithms** including:

* K-Nearest Neighbors (KNN)
* Linear Regression
* Logistic Regression
* Neural Networks (TensorFlow/Keras)

This project provides a **hands-on environment** for preprocessing data, training models, evaluating performance, and visualizing results using Python.

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

## ▶️ 3. Start Learning

Open the notebooks in the `notebooks/` folder and run cells using:

```
Shift + Enter
```

---

# 📁 Project Structure

```
├── 📓 notebooks/              # Jupyter notebooks for each ML task
│   ├── data_preprocessing.ipynb      # Task 1: Data cleaning & preparation
│   ├── linear_regression.ipynb       # Task 2: Linear Regression model
│   ├── knn_classifier.ipynb          # Task 3: KNN implementation
│   ├── logistic_regression.ipynb     # Binary classification
│   ├── neural_networks.ipynb         # Deep learning with Keras
│   └── visualization.ipynb           # Graphs & model evaluation
│
├── 🔧 scripts/                # Utility scripts
│   ├── launch_jupyter.ps1
│   ├── test_setup.py
│   └── verify_setup.py
│
├── 📊 data/                   # Datasets used for training/testing
├── 📁 outputs/                # Model outputs and visualizations
├── 🐍 .venv/                  # Virtual environment
├── 📄 requirements.txt        # Dependencies
└── 📄 .gitignore
```

---

# 📚 Tasks Covered

---

## 🔹 Task 1: Data Preprocessing

* Handle missing values (mean, median, drop)
* Encode categorical variables (One-hot, Label Encoding)
* Normalize / Standardize features
* Split dataset into training & testing sets

---

## 🔹 Task 2: Linear Regression

* Train regression model
* Interpret coefficients
* Evaluate using:

  * Mean Squared Error (MSE)
  * R-squared (R²)

---

## 🔹 Task 3: K-Nearest Neighbors (KNN)

* Train KNN classifier
* Experiment with different values of **K**
* Evaluate using:

  * Accuracy
  * Confusion Matrix
  * Precision & Recall

---

## 🔹 Task 4: Logistic Regression

* Binary classification model
* Interpret coefficients & odds ratio
* Evaluate using:

  * Accuracy
  * Precision & Recall
  * ROC Curve

---

## 🔹 Task 5: Neural Networks (TensorFlow/Keras)

* Build neural network architecture
* Train using backpropagation
* Evaluate performance
* Visualize training & validation loss

---

# ⚙️ Technologies Used

* **Python 3.x**
* **pandas** – Data handling
* **NumPy** – Numerical computations
* **scikit-learn** – Machine learning models
* **TensorFlow / Keras** – Deep learning
* **matplotlib** – Data visualization

---

# 📊 Example: Model Evaluation

```python
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

# 🧠 Key Learning Outcomes

By working on this project, you will understand:

* How to preprocess real-world datasets
* How different ML algorithms work
* How to evaluate model performance
* Differences between:

  * KNN (distance-based)
  * Regression models (linear)
  * Neural networks (non-linear)

---

# 🛠️ Troubleshooting

### ❌ "No module found"

👉 Activate the virtual environment and reinstall dependencies

### ❌ Jupyter not opening

👉 Run:

```powershell
.venv\Scripts\jupyter notebook
```

### ❌ Wrong kernel

👉 Select:

```
Python (your virtual environment)
```

---

# 🎯 How to Use This Repo

1. Start with **data_preprocessing.ipynb**
2. Move to **linear_regression.ipynb**
3. Try **knn_classifier.ipynb**
4. Explore **logistic_regression.ipynb**
5. Finish with **neural_networks.ipynb**

---

# 🎓 Disclaimer

This project is for **educational purposes only** and is designed to help understand machine learning concepts.

---

# 🎉 You're Ready!

* Run notebooks
* Experiment with models
* Try different datasets
* Improve your ML skills 🚀

---

**Happy Learning & Building! 🤖🔥**
