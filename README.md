# 🏆 Public Health Vaccine Uptake Prediction — Multi-Label Machine Learning Pipeline
### IIT Guwahati Data Analytics Hackathon · Flu Shot Learning Benchmark Challenge

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end multi-label probabilistic machine learning classification pipeline developed for the **IIT Guwahati Data Analytics Hackathon** (in collaboration with **GeeksforGeeks** and based on the **DrivenData Flu Shot Learning** challenge). The system models survey telemetry across 26,707 respondents to simultaneously predict the dual probabilities of receiving both the pandemic H1N1 vaccine (`xyz_vaccine`) and the seasonal influenza vaccine (`seasonal_vaccine`).

---

## 📌 Problem Formulation & Challenge Context

In public health epidemiology, immunization campaigns require granular forecasting of vaccine hesitancy and uptake across diverse demographics. This competition tasks data scientists with predicting two distinct, correlated binary outcomes for each individual:

1. **`xyz_vaccine`**: Whether the respondent received the H1N1 / pandemic influenza vaccine ($y_1 \in \{0, 1\}$).
2. **`seasonal_vaccine`**: Whether the respondent received the seasonal influenza vaccine ($y_2 \in \{0, 1\}$).

Because an individual's attitudes toward vaccination correlate across both vaccines, the problem is formulated as a **Multi-Output Probabilistic Classification** task rather than independent isolated models.

---

## 🎯 Evaluation Metric (Mean ROC-AUC)

Model performance is evaluated using the **mean Area Under the Receiver Operating Characteristic Curve (ROC-AUC)** averaged across both target labels:

$$\text{Score} = \frac{\text{ROC-AUC}(\text{xyz\_vaccine}) + \text{ROC-AUC}(\text{seasonal\_vaccine})}{2}$$

Each prediction must be a calibrated probability in the interval $[0.0, 1.0]$.

---

## 📊 Feature Taxonomy & Survey Feature Space

The dataset consists of **35 multi-modal predictor features** capturing behavioral habits, medical advice, subjective risk perceptions, and demographic backgrounds:

| Feature Category | Sample Features | Modeling Treatment |
| :--- | :--- | :--- |
| **Behavioral Habits** | `behavioral_antiviral_meds`, `behavioral_avoidance`, `behavioral_face_mask`, `behavioral_wash_hands`, `behavioral_large_gatherings`, `behavioral_outside_home`, `behavioral_touch_face` | Binary survey responses imputed with mode/most-frequent strategy. |
| **Medical Opinions** | `opinion_xyz_vacc_effective`, `opinion_xyz_risk`, `opinion_xyz_sick_from_vacc`, `opinion_seas_vacc_effective`, `opinion_seas_risk`, `opinion_seas_sick_from_vacc` | 1–5 Likert scale ordinal variables; scaled and standardized. |
| **Clinical Context** | `doctor_recc_xyz`, `doctor_recc_seasonal`, `chronic_med_condition`, `child_under_6_months`, `health_worker`, `health_insurance` | Critical predictive indicators with non-random missingness. |
| **Demographics** | `age_group`, `education`, `race`, `sex`, `income_poverty`, `marital_status`, `rent_or_own`, `employment_status`, `hhs_geo_region`, `census_msa` | Multi-category nominal attributes processed with `OneHotEncoder`. |

---

## 🏛️ Pipeline Architecture & Engineering

The solution is implemented in `Hackathon.ipynb` through a modular `scikit-learn` pipeline architecture:

```
                     ┌───────────────────────────┐
                     │ Raw Survey Features (X)   │
                     └─────────────┬─────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        ┌───────────────────────┐     ┌───────────────────────┐
        │ Numeric / Ordinal     │     │ Categorical Features  │
        │ - SimpleImputer       │     │ - SimpleImputer (mode)│
        │ - StandardScaler      │     │ - OneHotEncoder       │
        └───────────┬───────────┘     └───────────┬───────────┘
                    └──────────────┬──────────────┘
                                   ▼
                      ┌─────────────────────────┐
                      │    ColumnTransformer    │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │ MultiOutputClassifier   │
                      │ (Probabilistic Base)    │
                      └────────────┬────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
       ┌──────────────────────┐          ┌──────────────────────┐
       │ P(xyz_vaccine)       │          │ P(seasonal_vaccine)  │
       │ ROC-AUC Evaluator    │          │ ROC-AUC Evaluator    │
       └──────────────────────┘          └──────────────────────┘
```

### Key Engineering Decisions:
- **`ColumnTransformer` Integration**: Guarantees zero data leakage between training cross-validation folds and out-of-fold evaluation.
- **`MultiOutputClassifier`**: Wraps underlying binary estimators to handle the $(N \times 2)$ target label matrix while allowing independent calibrated probability thresholds for each target.
- **Diagnostic ROC Curves**: Built custom multi-panel diagnostic plotting (`plot_roc`) displaying ROC curves, optimal threshold trajectories, and individual vs. composite AUC scores.

---

## 📁 Repository Artifacts

- **`Hackathon.ipynb`**: Primary production notebook containing complete data ingestion, pre-processing transformers, multi-output model fitting, ROC evaluation, and final submission generation.
- **`Lab1_DS.ipynb`**: Auxiliary exploratory data science notebook testing feature correlation matrices and baseline classifiers.
- **`training_set_features.csv`**: In-sample feature dataset (26,707 rows, 36 columns).
- **`training_set_labels.csv`**: Ground-truth target labels for `xyz_vaccine` and `seasonal_vaccine`.
- **`test_set_features.csv`**: Out-of-sample feature set for competition scoring.
- **`submission_format.csv`**: Benchmark submission template for test set predictions.

---

## 🚀 Reproduction & Usage

```bash
# 1. Clone the repository
git clone https://github.com/Alokkr00/Hackathon.git
cd Hackathon

# 2. Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# 3. Execute the pipeline in Jupyter
jupyter notebook Hackathon.ipynb
```\n