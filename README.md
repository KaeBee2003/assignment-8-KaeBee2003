# Ensemble Learning for Bike Share Demand Prediction

## Overview
This project implements and compares three primary ensemble learning techniques (Bagging, Boosting, and Stacking) to predict bike rental demand using the UCI Bike Sharing Dataset. The analysis demonstrates how different ensemble methods address model variance and bias to achieve superior predictive performance.

## Dataset
- **Source**: UCI Machine Learning Repository - Bike Sharing Dataset
- **Citation**: Fanaee-T, Hadi, and Gamper, H. (2014). Bikeshare Data Set. UCI Machine Learning Repository.
- **Size**: 17,379 hourly samples
- **Target Variable**: `cnt` (total count of bike rentals)

## Project Structure
```
.
├── image.png             # Flow Chart Diagram for Stacking
├── hour.csv              # Dataset (download from UCI repository)
├── README.md             # This file
└── Ass8.ipynb            # Jupyter notebook with implementation
```

## Dependencies
```python
numpy
pandas
matplotlib
scikit-learn
```

Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Methodology

### Part A: Data Preprocessing & Baseline Models
1. **Feature Engineering**:
   - Dropped irrelevant columns: `instant`, `dteday`, `casual`, `registered`
   - One-hot encoded categorical features: `season`, `weathersit`, `mnth`, `hr`, `weekday`, `workingday`
   - Retained numerical features: `temp`, `atemp`, `hum`, `windspeed`

2. **Baseline Models**:
   - Decision Tree Regressor (max_depth=6): RMSE = 118.46
   - Linear Regression: RMSE = 100.45

### Part B: Ensemble Techniques

#### 1. Bagging (Variance Reduction)
- **Purpose**: Reduce prediction variance through bootstrap aggregation
- **Implementation**: 50 Decision Tree estimators
- **Result**: RMSE = 112.35 (improved from 118.46)
- **Key Finding**: Effectively reduced variance in high-variance models (Decision Trees), but minimal impact on low-variance models (Linear Regression: RMSE = 100.42)

#### 2. Boosting (Bias Reduction)
- **Purpose**: Sequentially reduce prediction bias
- **Implementation**: Gradient Boosting Regressor
- **Result**: RMSE = 78.97
- **Key Finding**: Significant improvement over both baseline models by learning from residual errors

### Part C: Stacking Regressor

#### Architecture
**Level-0 Learners (Base Models)**:
- K-Nearest Neighbors Regressor (captures local patterns)
- Bagging Regressor (variance reduction)
- Gradient Boosting Regressor (bias reduction)

**Level-1 Learner (Meta-Model)**:
- Ridge Regression (learns optimal combination weights)

#### How Stacking Works
1. Base models independently learn from training data
2. Their predictions become features for the meta-learner
3. Meta-learner optimally combines predictions to minimize error
4. Ridge regularization prevents overfitting in the combination

**Result**: RMSE = 67.05 (best performance)

## Results Summary

| Model | RMSE | Performance |
|-------|------|-------------|
| Decision Tree (Baseline) | 118.46 | Baseline |
| Linear Regression (Baseline) | 100.45 | Baseline |
| Bagging Regressor (DT) | 112.35 | 5.2% improvement |
| Bagging Regressor (LR) | 100.42 | Minimal change |
| Gradient Boosting | 78.97 | 21.4% improvement |
| **Stacking Regressor** | **67.05** | **33.2% improvement** |

## Key Insights

### Why Stacking Outperformed
1. **Model Diversity**: Combined models with different strengths and inductive biases
   - KNN: Captures local, instance-based patterns
   - Bagging: Reduces variance through aggregation
   - Boosting: Reduces bias through sequential learning

2. **Optimal Combination**: Meta-learner (Ridge) learned which base model to trust for different data patterns

3. **Bias-Variance Balance**: Achieved the best trade-off by leveraging complementary strengths of diverse models

### Theoretical Validation
- **Bagging**: Confirmed variance reduction in high-variance models (Decision Trees)
- **Boosting**: Demonstrated significant bias reduction through residual learning
- **Stacking**: Proved that ensemble diversity leads to superior generalization

## Visualizations
The notebook includes:
- Variance reduction comparison (Bagging vs single Decision Tree)
- Bias reduction demonstration (Boosting vs Linear Regression)
- Performance comparison across all models



## Conclusion
This experiment demonstrates that **no single model is universally superior**. However, by combining models that make different types of errors, stacking achieves the best bias-variance balance and delivers the strongest predictive performance (RMSE = 67.05, a 33.2% improvement over the baseline).

The results validate ensemble learning theory: diverse base learners + intelligent combination = superior performance.

## Author
**N V Karthik Balaji - DA25C014**
Assignment completed for DA5401: Data Analytics Lab
