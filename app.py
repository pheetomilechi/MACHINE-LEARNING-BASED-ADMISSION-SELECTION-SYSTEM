"""Web interface for the ML based admission selection system."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request
from waitress import serve


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model.pkl"
app = Flask(__name__)

FIELD_RULES = {
    "age": (14, 60, "Age"),
    "utme_score": (0, 400, "UTME score"),
    "post_utme_score": (0, 100, "Post-UTME score"),
    "olevel_points": (0, 45, "O'level points"),
    "subject_relevance": (0, 1, "Subject relevance"),
    "interview_score": (0, 100, "Interview score"),
}
OPTIONS = {
    "catchment_category": ["Catchment", "Non-Catchment", "ELDS"],
    "course_choice": ["First Choice", "Second Choice"],
}
DEFAULTS = {
    "age": "18", "utme_score": "250", "post_utme_score": "60",
    "olevel_points": "35", "subject_relevance": "0.80",
    "interview_score": "65", "catchment_category": "Catchment",
    "course_choice": "First Choice",
}


def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["model_name"]


def parse_candidate(form):
    candidate = {}
    errors = []
    for field, (minimum, maximum, label) in FIELD_RULES.items():
        try:
            value = float(form.get(field, "").strip())
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not minimum <= value <= maximum:
            errors.append(f"{label} must be between {minimum} and {maximum}.")
            continue
        candidate[field] = value
    for field, allowed_values in OPTIONS.items():
        value = form.get(field, "")
        if value not in allowed_values:
            errors.append(f"Choose a valid {field.replace('_', ' ')}.")
        else:
            candidate[field] = value
    return candidate, errors


@app.route("/", methods=["GET", "POST"])
def index():
    values = DEFAULTS.copy()
    result = None
    errors = []
    try:
        _, model_name = load_model()
        model_ready = True
    except FileNotFoundError:
        model_name = "Model unavailable"
        model_ready = False
        errors.append("No trained model found. Run train_model.py first.")
    if request.method == "POST":
        values.update(request.form.to_dict())
        candidate, errors = parse_candidate(request.form)
        if not errors and model_ready:
            pipeline, _ = load_model()
            frame = pd.DataFrame([candidate])
            prediction = int(pipeline.predict(frame)[0])
            probability = float(pipeline.predict_proba(frame)[0][1])
            result = {"admitted": prediction == 1,
                      "percentage": probability * 100}
    return render_template("index.html", values=values, result=result,
                           errors=errors, model_name=model_name,
                           model_ready=model_ready)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
