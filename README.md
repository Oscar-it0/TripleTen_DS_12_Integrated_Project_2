# Sprint 12 - Module 2 Project: Gold Recovery from Gold Ore - Zyfra

This project aims to build a machine learning model to predict gold recovery in a processing plant. We use real production data, including physical and chemical parameters measured at different stages of the process.

---

## Datasets

The data is contained in three CSV files:
- `gold_recovery_train.csv`: training set.
- `gold_recovery_test.csv`: test set (without targets).
- `gold_recovery_full.csv`: complete set with all features.

> The data is indexed by date and time (`date`). Parameters close in time are usually similar.

---

## Project Instructions

### 1. Data Preparation
- Load and explore the files from:
`gold_recovery_train.csv`
`gold_recovery_test.csv`
`gold_recovery_full.csv`
- Verify the calculation of `rougher.output.recovery` and compare with actual values using the Mean Absolute Error (MAE).
- Identify missing features in the test set and analyze their type.
- Perform the necessary preprocessing (null values, data types, etc.).

### 2. Exploratory Analysis
- Analyze how the concentrations of **Au**, **Ag**, and **Pb** change at each stage of the process.
- Compare the particle size distribution between the training and test sets.
- Evaluate the total sum of concentrations at each stage to detect anomalous values and decide whether they should be removed.

### 3. Model Construction
- Implement a function to calculate the **sMAPE** (Symmetric Mean Absolute Percentage Error).
- Train multiple models and evaluate them with cross-validation.
- Select the best model and test it with the test set.

---

## Checklist

- Quality of analysis and data preparation.
- Variety and performance of the models developed.
- Correct validation and evaluation of the model.
- Clarity in the explanation of each step.
- Cleanliness and organization of the code.
- Conclusions obtained.

---

## Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Pyplot
- Sklearn