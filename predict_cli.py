"""
predict_cli.py
----------------
Simple command-line interface that loads trained_model.pkl and predicts the
admission outcome for a candidate whose details are typed in at the prompt.
"""

import joblib
import pandas as pd

MODEL_PATH = "trained_model.pkl"


def get_float(prompt, lo=None, hi=None):
    while True:
        try:
            val = float(input(prompt))
            if lo is not None and val < lo:
                raise ValueError
            if hi is not None and val > hi:
                raise ValueError
            return val
        except ValueError:
            print(f"  Please enter a valid number{f' between {lo} and {hi}' if lo is not None else ''}.")


def get_choice(prompt, options):
    options_str = "/".join(options)
    while True:
        val = input(f"{prompt} ({options_str}): ").strip()
        for opt in options:
            if val.lower() == opt.lower():
                return opt
        print(f"  Please type one of: {options_str}")


def main():
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    model_name = bundle["model_name"]

    print("=" * 55)
    print(f" ML ADMISSION SELECTION SYSTEM  (model: {model_name})")
    print("=" * 55)

    candidate = {
        "age": get_float("Candidate age: ", 14, 60),
        "utme_score": get_float("UTME/JAMB score (0-400): ", 0, 400),
        "post_utme_score": get_float("Post-UTME score (0-100): ", 0, 100),
        "olevel_points": get_float("O'level aggregate points (0-45): ", 0, 45),
        "subject_relevance": get_float("Subject relevance to course (0-1): ", 0, 1),
        "interview_score": get_float("Interview score (0-100): ", 0, 100),
        "catchment_category": get_choice(
            "Catchment category", ["Catchment", "Non-Catchment", "ELDS"]
        ),
        "course_choice": get_choice(
            "Course choice", ["First Choice", "Second Choice"]
        ),
    }

    X_new = pd.DataFrame([candidate])
    prediction = pipeline.predict(X_new)[0]
    probability = pipeline.predict_proba(X_new)[0][1]

    print("\n" + "-" * 55)
    if prediction == 1:
        print(f"RESULT: ADMITTED  (confidence: {probability * 100:.1f}%)")
    else:
        print(f"RESULT: NOT ADMITTED  (admission likelihood: {probability * 100:.1f}%)")
    print("-" * 55)


if __name__ == "__main__":
    main()
