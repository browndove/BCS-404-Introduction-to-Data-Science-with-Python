# BCS 404 — Titanic Data Science Project

Exploratory Data Analysis, Statistical Analysis and Machine Learning on the **Titanic** dataset.

**Institution:** Accra Technical University  
**Department:** Computer Science  
**Course:** BCS 404 — Introduction to Data Science with Python  
**Academic Year:** 2025/2026 Second Semester  
**Lecturer:** Dr. Joseph Dadzie  

---

## Project summary

This repository contains a complete student project covering:

1. **Data acquisition** — load and inspect the Kaggle Titanic training data  
2. **Data cleaning** — missing values, duplicates, documented decisions  
3. **Visualisation** — histogram, bar chart, boxplot, scatter, heatmap, pairplot  
4. **Statistical analysis** — descriptive stats, frequencies, correlations  
5. **Machine learning** — Logistic Regression survival classifier with evaluation metrics  
6. **Discussion & conclusion** — findings, limitations, recommendations  

## Repository structure

```
datascience/
├── data/
│   └── train.csv                          # Titanic training dataset
├── notebooks/
│   └── BCS404_Titanic_EDA_ML.ipynb        # Main Jupyter Notebook (submit this)
├── src/
│   └── titanic_analysis.py                # Optional runnable Python script
├── figures/                               # Generated plots
├── reports/
│   ├── BCS404_Titanic_Project_Report.md   # Academic report (Markdown)
│   └── BCS404_Titanic_Project_Report.pdf  # Academic report (PDF)
├── requirements.txt
└── README.md
```

## Dataset

Source: [Kaggle Titanic competition data](https://www.kaggle.com/competitions/titanic/data)

The file `data/train.csv` is the standard Titanic training set (891 passengers × 12 columns).  
Kaggle competition data is redistributable for educational use under the competition rules; always credit Kaggle as the source.

## Software requirements

- Python 3.9+
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn, scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to run

### Option A — Jupyter Notebook (recommended)

```bash
cd notebooks
jupyter notebook BCS404_Titanic_EDA_ML.ipynb
```

Run all cells from top to bottom. Figures are saved to `../figures/`.

### Option B — Python script

```bash
python src/titanic_analysis.py
```

## Key results (reference)

| Item | Result |
|------|--------|
| Cleaned sample size | 891 passengers |
| Strongest positive correlation | SibSp–Parch (≈ 0.42) |
| Strongest negative correlation | Pclass–Fare (≈ −0.55) |
| Logistic Regression accuracy | ≈ **80.4%** |

## Report

- Print the PDF for the hardcopy submission.
- Softcopy: push this repository to GitHub and submit the URL as instructed by the lecturer.

## Submission checklist

- [x] Student name and index number on the report cover page and notebook header  
- [ ] Jupyter Notebook (`.ipynb`) in the repository  
- [ ] Dataset (`data/train.csv`)  
- [ ] Report PDF  
- [ ] This `README.md`  
- [ ] Printed hardcopy submitted to the Class Representative  
- [ ] GitHub URL submitted via the course platform  

**Deadline:** Sunday, 19 July 2026 at 11:59 p.m.

## Author

**Christian Ampeh Oduro** (01252798B) — Accra Technical University, Department of Computer Science
