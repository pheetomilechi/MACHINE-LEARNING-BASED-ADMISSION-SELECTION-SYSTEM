# Admissions Web App

A browser-based UI for the ML Based Admission Selection System, built with Flask.

## Folder structure
```
webapp/
├── app.py                 # Flask backend (loads trained_model.pkl, serves /evaluate API)
├── trained_model.pkl      # Trained pipeline (copy the latest one from train_model.py here if you retrain)
├── templates/
│   └── index.html         # The evaluation form page
└── static/
    ├── style.css          # Letterhead / ledger / stamp design
    └── script.js          # Submits the form via fetch() and animates the result stamp
```

## Setup

```bash
pip install flask joblib pandas scikit-learn
```

## Run

```bash
cd webapp
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## How it works
- The form on the page collects the same fields the model was trained on (age, UTME
  score, post-UTME score, O'level points, subject relevance, interview score,
  catchment category, course choice).
- On "Evaluate candidate", the page sends the data as JSON to `POST /evaluate`.
- The Flask backend validates the input, runs it through the saved scikit-learn
  pipeline, and returns `{admitted, confidence, model_name}`.
- The page displays the verdict as an animated stamp — green "Admitted" or
  rust "Not Admitted" — with the model's confidence percentage.

## Updating the model
If you retrain the model (`python train_model.py` in the parent folder), copy the new
`trained_model.pkl` into this `webapp/` folder and restart the Flask app.

## Deploying beyond localhost
The built-in Flask server (`app.run(...)`) is for development only. For a real
deployment, run it behind a production WSGI server such as **gunicorn** or **waitress**,
e.g.:
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```