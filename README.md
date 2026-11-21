# Project 2: Machine Learning for Engineering

## Predicting Concrete Compressive Strength Using MATLAB

**Student:** Muhammed L. Janko

## Project Overview
This project uses machine learning to predict the compressive strength of concrete based on its mixture ingredients and curing age.

## Dataset
The project uses the Concrete Compressive Strength dataset from the UCI Machine Learning Repository.

The dataset contains 8 input variables and 1 target variable, concrete compressive strength measured in MPa.

## Machine Learning Models
Three regression models were evaluated:

1. Linear Regression
2. Decision Tree Regression
3. Bagged Trees Regression

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 8.2576 | 10.3939 | 0.57162 |
| Decision Tree | 5.1974 | 7.0542 | 0.80268 |
| Bagged Trees | 4.2618 | 5.5523 | 0.87776 |

## Results
Bagged Trees achieved the best prediction performance, with an R² of approximately 0.878.

Curing age and cement content were identified as the most important predictors.

## Project Files
- `concrete_ml.m` — MATLAB source code
- `data/` — Concrete dataset
- `figures/` — Generated graphs
- `model_comparison.csv` — Model performance results
- `Project_2_Concrete_Strength_Final_Report.docx` — Project report

## How to Run
Open MATLAB, set the current folder to the project directory, and run `concrete_ml.m`.

The project requires MATLAB with Statistics and Machine Learning Toolbox.