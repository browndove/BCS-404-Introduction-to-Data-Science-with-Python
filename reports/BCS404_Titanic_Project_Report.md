# ACCRA TECHNICAL UNIVERSITY

**Department of Computer Science**

---

## BCS 404: Introduction to Data Science with Python

### PROJECT WORK

**Academic Year:** 2025/2026 Second Semester  
**Lecturer:** Dr. Joseph Dadzie

---

# From Deck Plans to Data: Exploring Survival on the Titanic

### EDA · Statistics · Machine Learning

### Dataset: Titanic Passenger Data (Kaggle)

---

**Student Name:** Christian Ampeh Oduro  

**Index Number:** 01252798B  

**Programme:** BSc Computer Science  

**Date of Submission:** July 2026

---

*Softcopy repository to be submitted via GitHub as required by the course brief.*

\newpage

# Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Description](#2-dataset-description)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Discussion](#5-discussion)
6. [Conclusion](#6-conclusion)
7. [References](#7-references)
8. [Appendix — Python Code](#8-appendix--python-code)

\newpage

# 1. Introduction

Data science combines statistical reasoning, programming and domain knowledge to extract insight from data. This project applies that workflow to a classic real-world dataset: passenger records from the RMS Titanic disaster of April 1912.

The aims of the study are to:

1. Acquire and inspect the Kaggle Titanic training dataset.
2. Clean missing and duplicated observations using justified decisions.
3. Produce and interpret key visualisations.
4. Conduct descriptive, frequency and correlation analyses.
5. Train a Logistic Regression classifier to predict passenger survival.
6. Discuss findings, limitations and recommendations.

Software used comprises **Python**, **Jupyter Notebook**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn** and **Scikit-Learn**, as specified in the course brief.

# 2. Dataset Description

The dataset was obtained from the Kaggle competition *Titanic — Machine Learning from Disaster*  
(https://www.kaggle.com/competitions/titanic/data).

| Attribute | Details |
|-----------|---------|
| File used | `train.csv` |
| Observations | 891 passengers |
| Variables | 12 original columns |
| Target | `Survived` (0 = No, 1 = Yes) |

**Variable overview**

| Variable | Description |
|----------|-------------|
| PassengerId | Unique passenger identifier |
| Survived | Survival outcome (target) |
| Pclass | Ticket class (1 = Upper, 2 = Middle, 3 = Lower) |
| Name | Passenger name |
| Sex | Sex (male/female) |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

# 3. Methodology

## 3.1 Data Acquisition
The CSV file was imported with `pandas.read_csv`. Dataset dimensions, column names, the first five rows and data types were inspected.

## 3.2 Data Cleaning
Missing values were detected with `isnull().sum()` and a missing-value heatmap.

| Column | Action | Justification |
|--------|--------|---------------|
| Age (~19.9% missing) | Median imputation | Continuous; median is robust to outliers |
| Embarked (2 missing) | Mode imputation | Categorical; very few cases |
| Cabin (~77.1% missing) | Column dropped | Too incomplete to impute reliably |
| Duplicates | Checked and removed if present | Preserve uniqueness |

No duplicate rows were found. After cleaning, the modelling table had **891 rows** and **11 columns** (Cabin removed) with **zero** remaining missing values.

## 3.3 Visualisation
Six plots were produced: age histogram; passenger-class bar chart; age-by-class boxplot; age-versus-fare scatter plot; correlation heatmap; and a pairplot of selected numerical variables. Each figure includes a title, axis labels and interpretation (see Results).

## 3.4 Statistical Analysis
Descriptive statistics, categorical frequency tables, and Pearson correlation analysis were computed. The strongest positive and negative pairwise correlations among numerical variables were identified.

## 3.5 Machine Learning
Predictor variables selected: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.  
Categorical predictors were label-encoded. The data were split **80% train / 20% test** with stratification on `Survived`. A **Logistic Regression** model (`max_iter=1000`) was trained and evaluated using accuracy, confusion matrix and classification report.

# 4. Results

## 4.1 Data Acquisition (Task 1)
- **Dimensions:** 891 rows × 12 columns  
- **Data types:** integers, floats and strings (object)  
- First five observations confirm mixed demographic and ticket fields with a binary survival flag.

## 4.2 Cleaning (Task 2)
Missingness was concentrated in `Cabin`, `Age` and (minimally) `Embarked`. After median/mode imputation and dropping `Cabin`, the cleaned frame was complete. Duplicate check: **0** duplicates.

## 4.3 Visualisations (Task 3)

**Histogram of ages.** Ages are right-skewed, with most passengers between roughly 20 and 40 years. A visible concentration near the median reflects median imputation of missing ages.

**Passenger class bar chart.** Third class is largest (491), then first (216), then second (184). Class imbalance is therefore both a demographic fact and a modelling consideration.

**Age by class boxplot.** Median age declines from class 1 to class 3: wealthier passengers tend to be older.

**Age versus fare scatter.** Fares are heavily right-skewed; a few high-fare outliers appear among older/first-class travellers. Survivors appear somewhat more often at higher fares.

**Correlation heatmap.** `Pclass` and `Fare` are strongly negatively associated. `Survived` associates negatively with `Pclass` and positively with `Fare`.

**Pairplot.** Pairwise plots reinforce that survival separates more clearly along class/fare than along age alone.

*(Figures are saved in the repository `figures/` folder and appear in the Jupyter Notebook.)*

## 4.4 Statistical Analysis (Task 4)

**Descriptive highlights (after cleaning):**
- Survival mean ≈ **0.384** (≈ 38.4% survived).
- Mean age ≈ **29.4** years (median = 28.0 after imputation).
- Mean fare ≈ **32.20**, with a large standard deviation (≈ 49.7), confirming skewness.

**Frequency distribution (selected):**
- Survived: 0 → 61.62%; 1 → 38.38%
- Sex: male 577; female 314
- Embarked: S dominant, then C, then Q

**Correlation findings:**
- Strongest **positive** pairwise correlation: **SibSp–Parch** (r ≈ **0.415**)
- Strongest **negative** pairwise correlation: **Pclass–Fare** (r ≈ **−0.550**)
- For survival specifically: `Pclass` (≈ −0.34) and `Fare` (≈ +0.26) are notable associations.

**Three important statistical findings**
1. Survival is a minority class (~38%); models must beat the ~62% “always die” baseline.
2. Socio-economic status (class and fare) is strongly linked to survival.
3. Sex differences and family co-travel (`SibSp`/`Parch`) are structurally important; females survive at much higher rates than males.

## 4.5 Machine Learning (Task 5)

| Metric | Result |
|--------|--------|
| Algorithm | Logistic Regression |
| Train / test size | 712 / 179 (stratified) |
| **Accuracy** | **≈ 0.804 (80.4%)** |

**Confusion matrix (test set):**

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| Actual 0 | 98 (TN) | 12 (FP) |
| Actual 1 | 23 (FN) | 46 (TP) |

**Classification report (approx.):**
- Class 0 (did not survive): precision ≈ 0.81, recall ≈ 0.89, F1 ≈ 0.85  
- Class 1 (survived): precision ≈ 0.79, recall ≈ 0.67, F1 ≈ 0.72  

**Coefficient directions:** male sex and higher class number (worse class) reduce predicted survival; higher fare slightly increases it; older age slightly decreases it. These signs align with historical “women and children first” and class-based access to lifeboats.

# 5. Discussion

*(Task 6 — approximately two pages of discussion.)*

## Major findings
The Titanic data exhibit clear structure: demographic (sex, age), socio-economic (class, fare) and family features all relate to survival. Visualisations and tables consistently show that women and higher-class passengers fared better. Third-class passengers dominate the sample count, yet they show lower survival rates than first-class passengers — a pattern that appears both in crosstabs and in the negative correlation between `Pclass` and `Survived`.

From a teaching perspective, the dataset is valuable because outcomes are historically documented and evacuation norms (“women and children first”) leave measurable traces in the tables. The project therefore links computation to social context rather than treating columns as abstract numbers only.

## Statistical insights
Pearson correlations quantify associations already visible in plots. Family counts move together (`SibSp`/`Parch`, r ≈ 0.42), while ticket class and fare move in opposite directions (r ≈ −0.55). Survival’s associations with class (negative) and fare (positive) provide statistical support for historically documented inequality of outcomes during the disaster.

Descriptive statistics further show that fare is heavily right-skewed: a small number of expensive tickets inflate the mean relative to the median. This skewness explains why median-based summaries and robust visual encodings (boxplots, scatter alpha) are more informative than means alone for fare.

Frequency analysis confirms class imbalance in the target: predicting “did not survive” for every passenger would already achieve about 62% accuracy. All model claims in this report are therefore judged relative to that baseline.

## Machine learning results
Logistic Regression is appropriate for an introductory binary classification task: it is interpretable and meets the brief. Achieving ~80% accuracy indicates that chosen predictors contain useful signal. However, **survivor recall (~0.67)** is weaker than non-survivor recall (~0.89), meaning the model still misses a non-trivial share of true survivors.

Coefficient signs are coherent with domain knowledge: encoded male sex and higher class number reduce log-odds of survival, while fare contributes a small positive association. This transparency is a pedagogical strength — students can explain *why* a prediction moved, not only that the accuracy number looks acceptable.

The default probability threshold of 0.5 may not maximise survivor recall. If the institutional goal were to identify as many survivors as possible, threshold tuning or class weights would be natural next steps, accompanied by precision–recall reporting rather than accuracy alone.

## Limitations
1. Median age imputation concentrates values and may understate age variability; the age histogram shows an artificial peak near the median.
2. Dropping `Cabin` discards deck location information that might relate to evacuation opportunity; cabin records were also more complete for first-class travellers, so missingness itself is informative.
3. `Name` and `Ticket` were unused; titles (Mr/Mrs/Master) often improve Titanic models.
4. Only one algorithm and one train–test split were evaluated; a different random seed or cross-validation folds could shift metrics slightly.
5. Analysis uses the competition training set only (n = 891), not the competition hold-out `test.csv` labels (unknown by design).

## Recommendations
1. Engineer title, family-size (`SibSp + Parch + 1`) and deck / “cabin known” features.
2. Compare Logistic Regression with Random Forest / Gradient Boosting under cross-validation.
3. Tune the classification threshold or class weights to improve survivor recall if that is the priority.
4. Report fairness metrics by sex and class for transparent evaluation.
5. Document all preprocessing in version-controlled notebooks for reproducibility and oral defence.

# 6. Conclusion

This project completed a full introductory data-science pipeline on the Kaggle Titanic dataset. After justified cleaning, visual and statistical analysis showed that sex and passenger class are central to survival patterns, while fare and family structure provide additional structure. A Logistic Regression classifier predicted survival at approximately **80%** test accuracy with coherent coefficient signs. Despite limitations in missing-data handling and feature engineering, the study demonstrates practical competence with Pandas, NumPy, Matplotlib, Seaborn and Scikit-Learn as required by BCS 404.

# 7. References

1. Accra Technical University, Department of Computer Science. *BCS 404: Introduction to Data Science with Python — Project Work Brief* (2025/2026 Second Semester).
2. Kaggle. *Titanic — Machine Learning from Disaster*. https://www.kaggle.com/competitions/titanic/data
3. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. (pandas)
4. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362.
5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.
6. Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

# 8. Appendix — Python Code

The complete executable analysis is provided in:

- `notebooks/BCS404_Titanic_EDA_ML.ipynb` (primary submission artefact)
- `src/titanic_analysis.py` (script equivalent)

Students should print selected notebook cells or attach the PDF export of the notebook as an appendix when preparing the hardcopy report.
