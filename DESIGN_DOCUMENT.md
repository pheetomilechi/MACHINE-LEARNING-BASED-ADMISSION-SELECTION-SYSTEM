# Design and Implementation of a Machine Learning Based Admission Selection System

## 1. Introduction
Manual admission processing in tertiary institutions is slow, inconsistent, and prone to
human bias. This project designs and implements a **Machine Learning Based Admission
Selection System (MLBASS)** that predicts whether a candidate should be **Admitted** or
**Rejected**, using historical admission data and supervised learning.

## 2. Aim and Objectives
**Aim:** To design and implement a system that uses machine learning to support fair,
fast, and consistent admission decisions.

**Objectives:**
1. Study the existing (manual) admission process and identify its weaknesses.
2. Design a dataset schema representing the key admission criteria.
3. Build a data preprocessing pipeline (cleaning, encoding, scaling).
4. Train and compare ML classification models to predict admission outcome.
5. Select the best-performing model and evaluate it with standard metrics.
6. Implement a simple front-end (GUI) that allows a registration officer to input a
   candidate's details and receive an instant recommendation.
7. Persist the trained model so it can be reused without retraining.

## 3. System Architecture

```
                +-------------------------------------------------+
                |                  PRESENTATION LAYER              |
                |   Tkinter Desktop GUI  /  CLI  /  (optional Web)  |
                +--------------------------+------------------------+
                                           |
                                           v
                +-------------------------------------------------+
                |                 APPLICATION LAYER                 |
                |  - Input validation                                |
                |  - Feature vector construction                    |
                |  - Calls the trained model for prediction          |
                +--------------------------+------------------------+
                                           |
                                           v
                +-------------------------------------------------+
                |                    ML LAYER                        |
                |  Preprocessing pipeline (encoders + scaler)         |
                |  Trained Classifier (Random Forest / Logistic /     |
                |  SVM / Gradient Boosting - best model auto-selected)|
                +--------------------------+------------------------+
                                           |
                                           v
                +-------------------------------------------------+
                |                   DATA LAYER                       |
                |  admission_dataset.csv (historical records)         |
                |  trained_model.pkl  (serialized model + pipeline)   |
                +-------------------------------------------------+
```

## 4. Dataset Design
Each record (candidate) has the following attributes:

| Feature              | Type        | Description                                      |
|----------------------|-------------|---------------------------------------------------|
| age                  | numeric     | Candidate's age                                    |
| utme_score           | numeric     | UTME/JAMB score (0-400)                            |
| post_utme_score      | numeric     | Post-UTME / screening score (0-100)                |
| olevel_points        | numeric     | Aggregated O'level grade points (best 5 subjects)  |
| subject_relevance    | numeric     | How relevant O'level subjects are to course (0-1)  |
| state_of_origin_cat  | categorical | Catchment / non-catchment / merit category         |
| course_choice        | categorical | First or second choice course                      |
| interview_score      | numeric     | Interview performance (0-100), where applicable    |
| admission_status     | target      | Admitted (1) / Rejected (0)                        |

In a real deployment this dataset would come from the institution's admissions database
(JAMB CAPS export, O'level result database, post-UTME records). For this implementation,
a **synthetic but realistic dataset** is generated programmatically so the full pipeline
can be demonstrated end-to-end.

## 5. Methodology
1. **Data Collection/Generation** – historical admission records.
2. **Data Preprocessing** – handle missing values, encode categorical variables
   (One-Hot Encoding), scale numeric features (StandardScaler).
3. **Train/Test Split** – 80/20 split, stratified on the target class.
4. **Model Training** – train and compare:
   - Logistic Regression (baseline)
   - Decision Tree
   - Random Forest
   - Gradient Boosting
5. **Model Evaluation** – Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC.
6. **Model Selection** – choose the model with the best F1/accuracy trade-off.
7. **Persistence** – save the winning pipeline (preprocessing + model) with `joblib`.
8. **Deployment (Demo)** – a Tkinter GUI loads the saved pipeline and predicts on new
   candidate data in real time.

## 6. Tools Used
- **Language:** Python 3
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, tkinter
- **IDE:** Any (VS Code, Jupyter, PyCharm)

## 7. File Structure
```
admission_system/
│── generate_dataset.py     # Creates the synthetic admission dataset (CSV)
│── admission_dataset.csv   # Generated dataset
│── train_model.py          # Preprocessing + training + evaluation + saving model
│── trained_model.pkl       # Serialized best pipeline (created after training)
│── model_comparison.png    # Bar chart comparing model accuracies
│── confusion_matrix.png    # Confusion matrix of the best model
│── predict_gui.py          # Tkinter GUI for making live predictions
│── predict_cli.py          # Command-line version of the predictor
│── DESIGN_DOCUMENT.md       # This document
```

## 8. Limitations and Future Work
- The demo dataset is synthetic; a production system must be retrained on verified
  institutional data and audited for bias (e.g., across gender/state of origin).
- Admission policies (quota, catchment area, affirmative action) should be encoded as
  business rules layered on top of the ML score, not left entirely to the model.
- Future work: add explainability (SHAP values) so admission officers can see *why*
  a candidate was recommended, and a web-based dashboard (Flask/Django) for multi-user
  access with authentication and audit logging.
