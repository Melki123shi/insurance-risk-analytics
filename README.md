# Insurance Risk Analytics & Predictive Modeling

**End-to-End Risk Analysis & Dynamic Pricing for Auto Insurance**  
**AlphaCare Insurance Solutions (ACIS)** — South Africa

---

## 📋 Project Overview

This project analyzes 18 months of historical car insurance data (Feb 2014 – Aug 2015) to optimize marketing strategy and develop **risk-based pricing models** for AlphaCare Insurance Solutions.

The goal is to identify low-risk customer segments, uncover key risk drivers across provinces, gender, and vehicle types, and build predictive models that estimate claim probability and severity — enabling smarter premium setting and improved profitability.

---

## 🎯 Business Objectives

- Discover **low-risk segments** for reduced premiums to attract new clients
- Statistically validate risk differences across provinces, zip codes, and gender
- Build predictive models for **Claim Severity** and **Claim Probability**
- Design a dynamic, data-driven pricing framework
- Provide actionable recommendations for marketing and product strategy

---

## 🛠️ Technologies Used

- **Python** 3.10
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn, XGBoost** – Machine Learning
- **SciPy, Statsmodels** – Statistical testing
- **SHAP** – Model interpretability
- **DVC** – Data Version Control
- **GitHub Actions** – CI/CD
- **Jupyter Notebooks**

---

## 📁 Project Structure

```bash
insurance-risk-analytics/
├── .github/workflows/      # CI/CD pipeline
├── data/                   # Data files (tracked by DVC)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
├── reports/
│   └── final_report.md
├── tests/
├── .dvc/                   # DVC configuration
├── dvc.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/insurance-risk-analytics.git
cd insurance-risk-analytics
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install DVC (for data versioning)
```bash
pip install dvc
dvc pull                     # Pull data from remote storage
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```

---

## 📊 Key Insights (Coming Soon)

- Overall portfolio Loss Ratio
- High-risk provinces and vehicle segments
- Gender-based risk differences
- Most influential features affecting claims (via SHAP)

---

## 📈 Analysis Workflow

1. **Task 1**: Exploratory Data Analysis (EDA)
2. **Task 2**: Data Version Control with DVC
3. **Task 3**: Statistical Hypothesis Testing (A/B Testing)
4. **Task 4**: Predictive Modeling & Risk-Based Pricing

---

## 🔄 Reproducing the Data Pipeline

### 1. Initialize DVC in the project
```bash
dvc init
```

### 2. Set up local remote storage
```bash
# Create a storage directory outside the project
mkdir  /path/to/local/storage 

# Add it as a DVC remote
dvc remote add -d myremote  /path/to/local/storage
```

### 3. Track the dataset with DVC
```bash
dvc add data/insurance_data.csv
git add data/insurance_data.csv.dvc data/.gitignore
git commit -m "Track insurance dataset with DVC"
```

### 4. Create multiple data versions
```bash
# Version 1: Raw data
cp data/insurance_data.csv data/insurance_data_raw.csv
dvc add data/insurance_data_raw.csv

# Version 2: Cleaned data
# (After running cleaning pipeline)
dvc add data/insurance_data_cleaned.csv
```

### 5. Push data versions to local remote
```bash
dvc push
```

### 6. To reproduce on another machine
```bash
dvc pull  # Restore all tracked data versions from remote storage
```

---

## 📝 Final Deliverables

- Full GitHub repository with CI/CD
- Reproducible DVC data pipeline
- Statistical validation of risk hypotheses
- Predictive models (Linear Regression, Random Forest, XGBoost)
- Professional final report (Medium-style)

---

## 👤 Author

**Melkishi**  
Marketing Analytics Engineer (Contract)  
AlphaCare Insurance Solutions

---

## 📌 Acknowledgments

- Data provided by AlphaCare Insurance Solutions
- Project guided by [Kerod, Mahbubah, Feven]

---

**Built with ❤️ for data-driven insurance innovation**
