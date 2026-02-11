# Meta-Learning for Automated Machine Learning Pipeline Selection

This project builds an automated machine learning (AutoML) pipeline that learns the best combination of preprocessing, feature selection, and classifier for predicting wine quality.

Instead of manually choosing a model, the system searches across an entire pipeline space and identifies the optimal configuration using cross-validated model selection.

Code reference: :contentReference[oaicite:0]{index=0}

📄 [Final Report (PDF)](report.pdf)  
🎞️ [Presentation Slides](slides.pdf)  

---

## Overview

The pipeline simultaneously learns:

- data scaling method
- feature selection strategy
- classifier choice
- hyperparameters

A full grid search is performed over 1320 candidate pipelines, using cross-validated accuracy as the objective.

This is a meta-learning approach: the system learns how to learn.

---

## Dataset

The dataset is the UCI Wine Quality dataset:

```
winequality-red.csv
```

- 1599 wine samples
- 11 continuous chemical features
- categorical quality score (3–8)

The task is multi-class classification.

---

## Pipeline Structure

The pipeline is composed of three learnable stages:

### 1. Data Scaling
- MinMaxScaler
- MaxAbsScaler
- RobustScaler

### 2. Feature Selection
- Variance Threshold
- SelectKBest

### 3. Classifiers
- Random Forest
- k-Nearest Neighbors
- Support Vector Machine
- AdaBoost

Each stage participates in the hyperparameter search.

---

## Method

A `PipelineHelper` object allows model families to be swapped inside a single sklearn pipeline. The full search space includes:

- scaling hyperparameters
- feature selection thresholds
- classifier hyperparameters

Grid search evaluates all combinations using 3-fold cross validation.

Baseline performance is compared against a default Random Forest model.


## Output

The program prints:

- baseline model accuracy
- ranked pipeline configurations
- cross-validation mean and variance
- optimal hyperparameter selections

This allows inspection of which pipeline structure performs best.

---

## Purpose

This project demonstrates:

- automated pipeline learning
- meta-model selection
- sklearn pipeline composition
- large-scale hyperparameter search
- practical AutoML experimentation

It is a research prototype for studying automated ML systems.

---

## Future Work

- Bayesian optimization instead of grid search
- nested cross validation
- neural meta-learners
- model ensembling
- dataset generalization tests
- runtime optimization

---

## Authors

Melissa Butler  
Emma Franz  
University of Wyoming
