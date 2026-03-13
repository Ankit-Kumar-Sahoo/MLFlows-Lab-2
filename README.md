# Coffee Quality Prediction — MLflow Lab

A modified version of the Wine Quality MLflow lab, adapted to predict **specialty-grade coffee** using sensory cupping data from the Coffee Quality Institute (CQI).

## What Changed from the Original Lab

### 1. Different Dataset
The original lab used the UCI Wine Quality dataset, combining red and white wine samples with physicochemical features like pH, sulphates, and alcohol content. This lab uses the **CQI Arabica Coffee dataset** (1,311 samples), where every feature is a sensory score assigned by a licensed Q-grader during a professional cupping session.

### 2. Different Features
Instead of chemical measurements, the 10 input features are the standard **SCA cupping attributes**: Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean Cup, Sweetness, and Cupper Points. These capture the human-perceived quality of a coffee rather than its lab-measured composition.

### 3. Different Target Definition
The original lab labeled a wine as high quality if its score was >= 7 out of 10. This lab uses the **SCA specialty threshold of 82 out of 100 Total Cup Points** as the cutoff — the industry-standard definition of specialty-grade coffee.

### 4. Stronger Baseline Model
The original used a Random Forest with only 10 estimators. This lab uses **100 estimators**, which produces a more reliable baseline before any hyperparameter tuning.

### 5. Additional EDA — Correlation Heatmap
On top of the box plots from the original, a **correlation heatmap** was added to check for multicollinearity among the sensory features, which is especially relevant given that cupping attributes often move together.

### 6. Additional Evaluation — Confusion Matrix & Classification Report
The original only reported AUC. This lab adds a **confusion matrix** and a full **classification report** (precision, recall, F1) on the test set, giving a clearer picture of how the model performs on each class.

---

## Dataset

**CQI Arabica Coffee Dataset** — 1,311 samples rated by licensed Q-graders.

Download and save to `data/arabica_data_cleaned.csv` before running the notebook:

```bash
mkdir data
curl -o data/arabica_data_cleaned.csv https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/arabica_data_cleaned.csv
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv coffee_env
source coffee_env/bin/activate        # Mac/Linux
coffee_env\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Notebook

```bash
jupyter notebook starter.ipynb
```

---

## Pipeline Steps

1. Load and explore the CQI Arabica dataset
2. Select 10 SCA sensory cupping attributes as features
3. Visualize Total Cup Points distribution; binarize target at threshold 82
4. Check for missing data
5. EDA: box plots per feature + correlation heatmap
6. 60/20/20 train/validation/test split
7. Train Random Forest (n=100); log params and AUC to MLflow
8. Feature importance analysis
9. Confusion matrix and classification report on test set
10. Register model as `coffee_quality` in MLflow Model Registry
11. Promote to Production stage
12. Load production model and run batch inference
13. Real-time inference via REST API — start the model server in a separate terminal first:
    ```bash
    mlflow models serve --env-manager=local -m models:/coffee_quality/production -h 0.0.0.0 -p 5001
    ```

---

## MLflow UI

```bash
mlflow ui
```

Open `http://localhost:5000` to view runs, metrics, and the model registry.

---

## Key Finding

`Cupper_Points`, `Flavor`, and `Balance` are the strongest predictors of specialty-grade coffee — consistent with the SCA's emphasis on overall impression and palate harmony.
