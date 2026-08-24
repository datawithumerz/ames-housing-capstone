\# Ames Housing Price Prediction



Exploratory data analysis of the Kaggle \*\*House Prices: Advanced Regression Techniques\*\* dataset, which contains information about residential properties in Ames, Iowa.



This project was completed as part of the \*\*CampusX Data Science Mentorship Program (DSMP 2.0)\*\*.



\## Project Overview



The main focus of this project was understanding and preparing the housing data through \*\*exploratory data analysis (EDA)\*\*.



The analysis covered:



\* Understanding the dataset and its features

\* Identifying numerical and categorical variables

\* Handling missing values

\* Examining distributions and skewness

\* Detecting and handling outliers

\* Analyzing relationships between features and `SalePrice`

\* Applying transformations to highly skewed variables

\* Examining correlations and potential multicollinearity



The project ends at the EDA and data-preparation stage. No regression model was built.



\## Dataset



\* \*\*Source:\*\* \[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

\* \*\*Training data:\*\* 1,460 observations

\* \*\*Features:\*\* 79 explanatory variables

\* \*\*Target:\*\* `SalePrice`

\* \*\*Feature types:\*\* Numerical and categorical



The dataset contains information about different aspects of residential properties, including size, quality, location, construction, garages, basements, and other property characteristics.



\## EDA Highlights



\### Target Variable



`SalePrice` was positively skewed, with an initial skewness of approximately \*\*1.88\*\*.



A `log1p` transformation reduced the skewness to approximately \*\*0.12\*\*, producing a much more balanced target distribution.



\### Feature Transformations



\* `SalePrice` was transformed using `log1p`.

\* `LotFrontage` was square-root transformed to reduce skewness.

\* Highly influential `GrLivArea` outliers with unusually low sale prices were identified and removed.



\### Correlation with SalePrice



Some of the strongest correlations with `SalePrice` were:



| Feature       | Correlation |

| ------------- | ----------: |

| `OverallQual` |        0.82 |

| `GrLivArea`   |        0.71 |

| `GarageCars`  |        0.68 |

| `TotalBsmtSF` |        0.64 |



These correlations provided an initial understanding of which variables were most strongly associated with house prices.



\### Multicollinearity



Several pairs of features showed strong relationships with each other, including:



\* `GarageArea` and `GarageCars`

\* `TotalBsmtSF` and `1stFlrSF`



These relationships were identified during the EDA as potential sources of multicollinearity.



\## Project Status



\*\*Completed: EDA and data preparation\*\*



The project does not include model training or evaluation. It represents the EDA stage of the Ames Housing dataset analysis.



\## Repository Structure



```ames-housing-capstone/

├── data/

│   ├── raw/

│   │   └── ames-hosuing.csv          

│   └── processed/

│       └── cleaned\_ames-housing.csv   

├── notebook.ipynb

├── README.md

└── .gitignore ...

```



