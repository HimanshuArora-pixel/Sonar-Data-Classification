# Sonar Data Classification: Neural Network Baseline vs. XGBoost Optimization

## 📌 Project Overview
This project tackles the classic **Rocks vs. Mines** binary classification problem. The primary engineering challenge was combating the **Curse of Dimensionality** and severe high variance caused by an extreme feature-to-sample ratio. 

The repository demonstrates a two-pronged approach to model stabilization and feature engineering:
1. **`Neural_Network_Optimization`**: Establishing a baseline using a Deep Learning architecture.
2. **`XGBoost_optimization`**: Pivoting to an advanced, GPU-accelerated gradient boosting strategy specifically designed to dominate small, high-dimensional tabular datasets.

## 📊 Dataset & The Challenge
* **Domain:** Sonar Target Detection
* **Dataset Size:** ~207 samples
* **Dimensionality:** 60 continuous features
* **Target Variable (Y):** Binary classification — mapped from `R` (Rock) and `M` (Mine) to numeric binaries (`0` and `1`).
* **The Core Challenge:** A 60:200 feature-to-sample ratio makes models highly susceptible to stochastic noise, resulting in severe overfitting and erratic validation accuracy swings.

---

## ⚙️ Engineering Lifecycle: Part 1 - The ANN Baseline (`Neural_Network_Optimization`)

### Diagnosing Variance & Taming the Data
* **Structural Upgrade:** Transitioned the model to a robust `from_logits=True` architecture using a `Dense(units=2, activation='linear')` output layer paired with `SparseCategoricalCrossentropy` for numerical stability.
* **Capacity Tuning:** Systematically tested various hidden layer topologies. Learning curve analysis revealed that larger networks immediately memorized the dataset (hitting 100% training accuracy) while validation performance flatlined. Selected a minimal `(15, 8)` architecture to artificially constrain the model.
* **Reproducibility Lockdown:** Initial testing showed validation accuracy swinging violently. Engineered a fully deterministic environment by implementing full dataset pre-shuffling (`df.sample(frac=1)`), enforcing strict class balance (`stratify=Y`), and locking global randomness across Numpy and TensorFlow.
* **Baseline Results:** * **Training Accuracy:** 92.73%
  * **Test Accuracy:** 78.57%
  * **Takeaway:** While powerful, the ANN struggled to generalize on the small tabular dataset, exhibiting slight overfitting.

---

## 🚀 Engineering Lifecycle: Part 2 - The Pivot (`XGBoost_optimization`)

To bridge the generalization gap, the strategy shifted to **XGBoost**, utilizing a committee of decision trees strictly evaluated on an 80/20 Train-Test split of unseen data.

### Stepwise Tuning & Hardware Acceleration
Rather than throwing algorithms blindly at the data, a highly analytical Stepwise Tuning (Coordinate Descent) approach was used to isolate and optimize hyperparameters:
* **Tree Complexity (`max_depth`):** Discovered that deep trees caused massive overfitting. Capping depth at `2` or `3` forced the model to stay shallow and focus only on the highest-signal features.
* **Forest Size (`n_estimators`):** Tested large forests (up to 2000 trees) to ensure the model had the capacity to learn the 60-dimensional space.
* **Step Size (`learning_rate`):** Adjusted incrementally (from `0.1` down to `0.005`) to find the perfect balance between learning speed and accuracy.

## 🏆 The Champion Model & Final Results

After extensive stepwise tuning, the optimal XGBoost model was locked in with the following architecture:
* `n_estimators`: **1250**
* `max_depth`: **3**
* `learning_rate`: **0.01**

### Performance Comparison

| Metric | ANN Baseline | XGBoost Champion | Improvement |
| :--- | :--- | :--- | :--- |
| **Training Accuracy** | 92.73% | 100.00% | +7.27% |
| **Test Accuracy** | 78.57% | **90.48%** | **+11.91%** |

The transition to a finely tuned, hardware-accelerated XGBoost model resulted in a massive ~12% jump in test accuracy, successfully overcoming the curse of dimensionality.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** Pandas, NumPy
* **Environment:** Jupyter Notebook
## 👨‍💻 Author
**Himanshu Arora**
* B.Tech Information Technology (Class of 2029) @ IIIT Allahabad
* Member, Club of Programmers (AI/ML Wing)
