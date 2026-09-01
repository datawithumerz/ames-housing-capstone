<div align="center">

# 🏠 Ames Housing Price Prediction

### End-to-End Regression, Feature Selection & Statistical Model Validation

*A disciplined, statistically defensible approach to regression modeling — not just a leaderboard score.*

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Diagnostics-8A2BE2)](https://www.statsmodels.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()
[![License](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

[Kaggle](https://www.kaggle.com/datawithumerz) · [GitHub](https://github.com/datawithumerz) · [LinkedIn](https://www.linkedin.com/in/datawithumerz/)

</div>

---

## 📌 Overview

An end-to-end regression project built on the Kaggle **[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)** dataset, using residential property data from Ames, Iowa.

Rather than optimizing purely for predictive performance, this project investigates **how to build a lean, interpretable, and statistically defensible regression model** — through systematic preprocessing, rigorous feature selection, formal regression diagnostics, and regularization.

> Completed as part of the **CampusX Data Science Mentorship Program (DSMP 2.0)**.

---

## 🏆 Key Results

| Stage                               |                    Result |
| ------------------------------------ | -------------------------: |
| Initial preprocessed feature space   |              **206 features** |
| After filter-based selection         |              **133 features** |
| After Sequential Forward Selection   |               **50 features** |
| After Lasso-based selection          |               **22 features** |
| **Final feature set**                |           **21 features** |
| **Final model**                      |      **Ridge Regression** |
| Target transformation                |    **`log1p(SalePrice)`** |
| Best cross-validated R²              |     **0.8910 ± 0.0165** |
| Held-out Test R²                     |                **0.8702** |
| Held-out Adjusted R²                 |                **0.8601** |

The final model is a **Ridge Regression**, tuned via `RidgeCV`, trained on the final **21-feature** dataset — selected after benchmarking OLS, Ridge, Lasso, and ElasticNet on both a held-out test set and 5-fold cross-validation.

---

## 🗺️ Project Workflow

```text
Raw Ames Housing Data
        │
        ▼
Exploratory Data Analysis
        ├── Missing-value analysis
        ├── Distribution & skewness analysis
        ├── Outlier investigation
        ├── Correlation analysis
        └── Target transformation
        │
        ▼
Data Preprocessing
        ├── Numerical scaling
        ├── Ordinal encoding
        └── One-hot encoding
        │
        ▼
206 Engineered Features
        │
        ▼
Filter-Based Selection ───────► 133 features
        │
        ▼
Wrapper-Based Selection (SFS) ─► 50 features
        │
        ▼
Embedded Selection (Lasso) ────► 22 features
        │
        ▼
Regression Diagnostics
        ├── Linearity
        ├── Independence
        ├── Homoscedasticity
        ├── Residual normality
        ├── Multicollinearity
        └── Influential observations
        │
        ▼
21 Final Features
        │
        ▼
OLS  vs  Ridge  vs  Lasso  vs  ElasticNet
        │
        ▼
✅ Final Ridge Regression Model
```

---

## 📓 Notebooks

| Notebook | Description |
| --- | --- |
| [`eda-ames-housing.ipynb`](./eda-ames-housing.ipynb) | Dataset understanding, missing values, distributions, skewness, outliers, transformations, correlations, and multicollinearity. |
| [`regression-modeling-ames-housing.ipynb`](./regression-modeling-ames-housing.ipynb) | Preprocessing, filter/wrapper/embedded feature selection, regression diagnostics, regularization, cross-validation, and final model selection. |

---

## 📊 Dataset

| Property | Value |
| --- | ---: |
| Source | [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| Observations | **1,460** |
| Explanatory variables | **79** |
| Target | **`SalePrice`** |
| Feature types | Numerical + categorical |
| Location | Ames, Iowa |

Property attributes include overall quality, living area, basement and garage characteristics, lot size, construction materials, neighborhood, year built/remodeled, room/bathroom counts, and property condition.

📥 **[Download the processed dataset (`clean-ames-housing.csv`)](https://raw.githubusercontent.com/datawithumerz/ames-housing-capstone/refs/heads/main/data/processed/clean-ames-housing.csv)**

---

## 🔍 Exploratory Data Analysis

### Target Transformation

The raw `SalePrice` distribution was strongly right-skewed. A `log1p` transformation brought it much closer to normal:

| Metric | Before | After `log1p` |
| --- | ---: | ---: |
| Skewness | **1.88** | **0.12** |

```python
y = np.log1p(df["SalePrice"])
```

### Feature Transformations

* `SalePrice` → `log1p`
* `LotFrontage` → square-root transformation
* Highly influential `GrLivArea` outliers with abnormally low sale prices were identified and removed

All transformations were driven by distributional evidence rather than applied by default.

### Strongest Correlations with `SalePrice`

| Feature | Correlation |
| --- | ---: |
| `OverallQual` | **0.82** |
| `GrLivArea`   | **0.71** |
| `GarageCars`  | **0.68** |
| `TotalBsmtSF` | **0.64** |

### Multicollinearity Signals

* `GarageArea` ↔ `GarageCars`
* `TotalBsmtSF` ↔ `1stFlrSF`

These relationships were flagged early and re-examined formally during the regression diagnostics stage.

---

## 🧩 Feature Selection

Preprocessing (scaling + ordinal encoding + one-hot encoding) expanded the cleaned dataset to **206 engineered features**. Instead of fitting a model directly to all 206, three complementary selection strategies were layered together:

| Stage | Method | Features Remaining |
| --- | --- | ---: |
| Preprocessing | Scaling + Encoding | **206** |
| Filter | Duplicate check + correlation pruning + `f_regression` | **133** |
| Wrapper | Sequential Forward Selection | **50** |
| Embedded | Lasso | **22** |
| Statistical refinement | OLS diagnostics | **21** |

**~90% reduction in feature space**, while retaining a compact, informative predictor set.

<details>
<summary><b>Why use three different selection methods?</b></summary>
<br>

| Method | Perspective |
| --- | --- |
| **Filter** | Removes redundant or weakly associated variables *before* model fitting. |
| **Wrapper** | Evaluates feature subsets based on actual model performance. |
| **Embedded** | Performs selection *during* fitting — Lasso shrinks some coefficients to exactly zero. |

Combining all three produces a far more deliberate selection process than relying on any single technique alone.

</details>

---

## 🩺 Regression Diagnostics

Before finalizing the model, the classical OLS assumptions were explicitly tested rather than assumed.

| Assumption | Finding |
| --- | --- |
| Linearity | ✅ Reasonably supported |
| Independence | ✅ Supported |
| Multicollinearity | ✅ Not a major concern (condition number **7.53** — well-conditioned) |
| Homoscedasticity | ⚠️ Heteroscedasticity detected |
| Residual normality | ⚠️ Some violation observed |
| Influential observations | 🔍 Investigated, not discarded |

Violations were treated as **documented model limitations**, not hidden issues — robust standard errors were applied where appropriate, and the relatively large sample size was factored into interpreting the residual-normality result.

---

## ⚖️ Model Comparison

Four linear models were evaluated on the final feature set using a held-out test set, Test R², Test Adjusted R², and 5-fold cross-validation:

| Model          |    Test R² | Test Adj. R² | CV R² Mean | CV R² Std |
| -------------- | ---------: | ------------: | ---------: | --------: |
| OLS            |     0.8698 |        0.8597 |     0.8909 |    0.0166 |
| Lasso          | **0.8708** |    **0.8607** |     0.8870 |    0.0171 |
| **Ridge** ⭐     |     0.8702 |        0.8601 | **0.8910** | **0.0165** |
| ElasticNet     |     0.8701 |        0.8599 |     0.8909 |    0.0166 |

**All four models performed extremely similarly** — and that's a meaningful result, not a disappointing one. By the time regularization entered the picture, the preceding **Filter → Wrapper → Embedded** pipeline had already removed most feature redundancy, leaving little multicollinearity for regularization to meaningfully exploit. The low condition number (**7.53**) corroborates this.

---

## 🎯 Why Ridge?

Ridge was selected not because it dramatically outperformed the alternatives — it didn't — but because it was the **right fit for this stage of the pipeline**.

**1. Strongest cross-validated performance**
Ridge achieved **CV R² = 0.8910 ± 0.0165** — the highest mean *and* the lowest variance across folds — without sacrificing held-out performance.

**2. Feature elimination was already handled**
Lasso had already served as the embedded feature-selection method, cutting the set to 22 variables. Using Lasso again as the final estimator would just add another, redundant round of elimination. At this stage, the goal shifted from *selection* to **stable estimation on an already-selected feature set** — exactly what Ridge is built for.

**3. Added coefficient stability at no cost**
Even with multicollinearity largely resolved, Ridge still guards against residual correlation among predictors — with essentially zero performance trade-off.

**4. ElasticNet added little marginal value**
ElasticNet's L1 component duplicates work the pipeline had already done explicitly. Since it performed nearly identically to Ridge, Ridge was preferred for its simpler, more direct formulation.

---

## ✅ Final Model

> **Ridge Regression**, trained on **21 selected features**, with `alpha` tuned via `RidgeCV`.

```text
206 engineered features
        ↓  Filter selection
133 features
        ↓  Sequential Forward Selection
50 features
        ↓  Lasso (embedded selection)
22 features
        ↓  OLS diagnostics (removed HalfBath — statistically insignificant)
21 final features
```

### Final Performance

| Metric | Score |
| --- | ---: |
| Held-out Test R² | **0.8702** |
| Held-out Adjusted R² | **0.8601** |
| 5-Fold CV R² | **0.8910 ± 0.0165** |

The small gap between test and cross-validated performance indicates the final model generalizes consistently across validation folds rather than overfitting to a single split.

---

## 🧠 What This Project Demonstrates

<table>
<tr>
<td valign="top" width="50%">

**Data Understanding**
- Exploratory data analysis
- Numerical & categorical feature analysis
- Missing-value investigation
- Distribution & skewness analysis
- Outlier detection
- Correlation analysis

**Feature Engineering & Preprocessing**
- Target transformation
- Numerical scaling
- Ordinal encoding
- One-hot encoding
- `ColumnTransformer`
- High-dimensional feature-space management

</td>
<td valign="top" width="50%">

**Feature Selection**
- Duplicate detection
- Correlation-based pruning
- `f_regression`
- Sequential Forward Selection
- Lasso / `LassoCV`

**Statistical Modeling & Validation**
- Ordinary Least Squares, Adjusted R²
- Residual, homoscedasticity & normality diagnostics
- Multicollinearity & influential-observation analysis
- Robust standard errors
- Ridge, Lasso, ElasticNet, `RidgeCV`
- 5-fold cross-validation & held-out evaluation

</td>
</tr>
</table>

Most importantly, this project demonstrates a **reasoned modeling workflow** — every major decision was evaluated against evidence, not applied as a checklist.

---

## 💡 Key Takeaways

1. **Feature selection mattered more than the final regularizer.** The biggest complexity reduction happened *before* the final model comparison: `206 → 133 → 50 → 22 → 21`.
2. **More regularization ≠ automatically better performance.** All four final models scored almost identically once the feature space was carefully controlled.
3. **Statistical diagnostics are part of model building**, not an afterthought — assumptions were tested, violations identified, and their implications weighed into the final decision.
4. **Model selection is about more than the top test score.** Lasso edged out on held-out R² (`0.8708`), but Ridge won on cross-validated stability (`0.8910`) — the final call weighed generalization, stability, and structural fit, not a single metric.

---

## 📁 Repository Structure

```text
ames-housing-capstone/
│
├── data/
│   ├── raw/
│   │   └── ames-housing.csv
│   │
│   └── processed/
│       └── clean-ames-housing.csv
│
├── eda-ames-housing.ipynb
├── regression-modeling-ames-housing.ipynb
├── README.md
└── .gitignore
```

---

## 🛠️ Tools & Libraries

<div align="center">

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `Statsmodels` · `Jupyter Notebook`

</div>

---

## 📌 Project Status

**✅ Completed** — the full workflow from exploratory analysis and preprocessing through feature selection, regression diagnostics, regularization, cross-validation, and final model selection is done.

---

<div align="center">

## 👤 Author

**Muhammad Umer Asad**
Software Engineering student focused on **Data Science, Machine Learning, and AI Engineering**.

[Kaggle](https://www.kaggle.com/datawithumerz) · [GitHub](https://github.com/datawithumerz) · [LinkedIn](https://www.linkedin.com/in/datawithumerz/)

</div>
