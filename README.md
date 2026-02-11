# ML Regression Model Comparison for Spatial Saturation Profiles

This project compares several regression approaches for modeling spatial saturation data. The notebook evaluates how well different machine learning models approximate a 1-D profile and analyzes training vs. test error across model complexity.

The goal is to understand bias–variance tradeoffs and model behavior using a controlled dataset.

Presentation: https://youtu.be/hxNFxsIoxms
[Final Report (PDF)](report.pdf)

---

## Dataset

The input data is loaded from:

```
data_profile_mean.txt
```

It represents a spatial saturation profile used as a regression target. The notebook assumes the data file is located in:

```
./data_profile_mean.txt
```

You may replace this with your own dataset in the same format.

---

## Methods Compared

The notebook implements and evaluates:

- Polynomial Regression  
- Decision Tree Regression  
- k-Nearest Neighbors Regression  
- Random Forest Regression  

Each model is trained across a range of hyperparameters and evaluated using mean squared error (MSE) on training and test splits.

---

## Features

- Automatic train/test split  
- Hyperparameter sweeps for each model  
- Training vs. test error visualization  
- Side-by-side model comparison  
- Clean plotting utilities for spatial interpretation  

---
