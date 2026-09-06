"""
generate_dataset.py
--------------------
Generates a synthetic (but realistic) admission dataset and saves it as
admission_dataset.csv. Replace this with a loader for your real institutional
data (e.g., pd.read_csv("real_admissions.csv")) when deploying for real use.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # number of candidate records

# ---- Simulate raw features -------------------------------------------------
age = np.random.randint(16, 25, N)
utme_score = np.clip(np.random.normal(220, 45, N), 100, 400).round(0)
post_utme_score = np.clip(np.random.normal(55, 15, N), 0, 100).round(1)
olevel_points = np.clip(np.random.normal(35, 6, N), 10, 45).round(0)  # higher = better (5 subjects, A1=45 ... F=0 scale)
subject_relevance = np.clip(np.random.normal(0.7, 0.2, N), 0, 1).round(2)
interview_score = np.clip(np.random.normal(60, 20, N), 0, 100).round(1)

catchment_options = ["Catchment", "Non-Catchment", "ELDS"]
catchment = np.random.choice(catchment_options, N, p=[0.4, 0.5, 0.1])

course_choice_options = ["First Choice", "Second Choice"]
course_choice = np.random.choice(course_choice_options, N, p=[0.75, 0.25])

# ---- Simulate the admission decision (label) --------------------------------
# We build a weighted "merit score" and add noise, then threshold it.
catchment_bonus = pd.Series(catchment).map({"Catchment": 8, "ELDS": 5, "Non-Catchment": 0}).to_numpy()
choice_bonus = pd.Series(course_choice).map({"First Choice": 5, "Second Choice": -5}).to_numpy()

merit_score = (
    0.30 * (utme_score / 400 * 100) +
    0.20 * post_utme_score +
    0.20 * (olevel_points / 45 * 100) +
    0.10 * (subject_relevance * 100) +
    0.10 * interview_score +
    0.10 * (catchment_bonus + choice_bonus) +
    np.random.normal(0, 5, N)  # noise
)

# Top ~45% (by merit score, with some randomness already baked in) get admitted
threshold = np.percentile(merit_score, 55)
admission_status = (merit_score >= threshold).astype(int)

df = pd.DataFrame({
    "age": age,
    "utme_score": utme_score,
    "post_utme_score": post_utme_score,
    "olevel_points": olevel_points,
    "subject_relevance": subject_relevance,
    "catchment_category": catchment,
    "course_choice": course_choice,
    "interview_score": interview_score,
    "admission_status": admission_status,
})

df.to_csv("admission_dataset.csv", index=False)
print(f"Generated {N} records -> admission_dataset.csv")
print(df["admission_status"].value_counts(normalize=True).rename("proportion"))
print(df.head())
