# Student Performance Prediction Using Machine Learning

An educational machine-learning project that estimates a student's final
mathematics grade (`G3`, 0-20) from the UCI Student Performance Dataset. It
compares Linear Regression, Decision Tree, and Random Forest models and shows
predictions, evaluation metrics, and exploratory charts in a Streamlit app.

> This is a learning demonstration. A prediction is an estimate, not a
> judgement about a student or a substitute for support from teachers.

## Problem statement

Schools and students may want to understand how earlier grades, study time,
attendance, and other factors relate to final results. This project demonstrates
how a regression workflow can explore those patterns and make a cautious
estimate from historical data.

## Objectives

- Load, inspect, and clean the dataset using Pandas and NumPy.
- Explore relationships in the data with Matplotlib and Seaborn.
- Train and compare three regression algorithms.
- Select a model using training-only cross-validation, then report held-out
  test metrics.
- Provide an interactive Streamlit form for a single student's details.

## Technologies

Python 3.10 or newer, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib,
Streamlit, and pytest.

## Dataset

The project uses `student-mat.csv` from the
[UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance).
The CSV is semicolon-separated and contains 395 mathematics student records.
The target is `G3`, the final grade from 0 to 20. `G1` and `G2` are the first-
and second-period grades and are deliberately included as earlier assessment
features, as requested. Because they are strongly related to the final grade,
this prediction setup assumes those earlier grades are already known.

The official UCI archive is downloaded into `dataset/` for this project. If
automatic downloading is unavailable, download the dataset archive from UCI,
extract `student-mat.csv`, and put it here:

```text
dataset/student-mat.csv
```

## Project architecture

```text
student_performance_prediction/
├── dataset/
│   └── student-mat.csv
├── data/
│   └── processed/                 # cleaned CSV created by preprocessing
├── models/
│   └── best_model.joblib          # fitted preprocessing + regression pipeline
├── outputs/
│   ├── model_comparison.csv
│   └── visualizations/
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── train_model.py
├── tests/
│   └── test_pipeline.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Data preparation and leakage prevention

The preprocessing module checks required columns, missing values, duplicate
rows, and missing target grades. Exact duplicate rows and rows without `G3`
are removed. Numeric missing values are imputed with the training-fold median;
categorical missing values are imputed with the training-fold most frequent
category. Categorical fields are one-hot encoded and numeric fields are
standardized.

All imputers, encoders, and scalers live inside a Scikit-learn `Pipeline`.
Therefore, during cross-validation each transformation is fitted only on that
fold's training portion. A final 20% test set is kept out of model selection.

## Exploratory data analysis

The EDA script saves charts for grade distribution, study time, absences,
previous grades, gender category, parental education, and numeric correlations
under `outputs/visualizations/`.

## Models and evaluation

The training script compares:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Each model is tuned using 5-fold shuffled cross-validation on the training
split. The selected model is the one with the lowest mean cross-validation MAE;
the held-out test scores do **not** choose the model. The CSV reports CV MAE
and test MAE, MSE, RMSE, and R-squared for each model.

## Installation and execution in VS Code

1. Open the `student_performance_prediction` folder in VS Code.
2. Open **Terminal > New Terminal**.
3. Create and activate a virtual environment in PowerShell:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If activation is blocked by your PowerShell policy, use the environment's
   Python executable directly as shown below.

4. Install project dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. In VS Code, open the Command Palette (`Ctrl+Shift+P`), choose **Python:
   Select Interpreter**, and select `.venv`.
6. From the project folder, run data inspection and cleaning:

   ```powershell
   python -m src.data_preprocessing
   ```

7. Create and save the exploratory charts:

   ```powershell
   python -m src.eda
   ```

8. Train, compare, and save the best pipeline:

   ```powershell
   python -m src.train_model
   ```

9. Run the tests:

   ```powershell
   python -m pytest
   ```

10. Start the dashboard:

    ```powershell
    python -m streamlit run app.py
    ```

    Streamlit prints the local address; by default it is
    `http://localhost:8501`.

If you did not activate the virtual environment, prefix each command with
`.\.venv\Scripts\python.exe` instead of `python`.

## Reading the evaluation metrics

- **MAE**: average absolute grade error; smaller is better.
- **MSE**: average squared error; smaller is better and large errors count more.
- **RMSE**: square root of MSE, expressed in grade points; smaller is better.
- **R-squared**: proportion of target variation explained by the model; closer
  to 1 is generally better, but a negative value is possible.

Scores depend on the fixed random split, library versions, and selected
cross-validation parameters. The generated `outputs/model_comparison.csv`
contains the actual scores for the current run.

## Application screenshots

After launching the app, save screenshots here if desired:

```text
outputs/screenshots/
```

Screenshots are not generated automatically.

## GitHub preparation

The `.gitignore` excludes the virtual environment, Python bytecode, pytest
cache, and Streamlit secrets. To publish this project:

```powershell
git init
git add .
git commit -m "Build student performance prediction project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```

Create an empty repository on GitHub first and replace the example remote URL
with your repository URL.

## Future enhancements

- Compare additional models and explainability methods.
- Add confidence intervals or prediction ranges.
- Add a model card and document limitations of the dataset.
- Allow uploading a batch of student rows for analysis.
- Add deployment and automated testing through GitHub Actions.