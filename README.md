# ML Admission Selection System

A Flask web application that loads the trained scikit-learn pipeline and evaluates admission candidates.

## Run locally

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

For local production-style serving with Gunicorn on Linux/macOS:

```bash
gunicorn --workers 2 --bind 0.0.0.0:5000 app:app
```

On Windows, the entry point uses Waitress:

```powershell
python app.py
```

## Deploy on Render

1. Push this entire project folder to a GitHub repository. Make sure `trained_model.pkl`, `templates/`, and `static/` are committed. Do not commit passwords, API keys, or other secrets.
2. In the Render dashboard, select **New +** -> **Web Service**.
3. Connect the GitHub repository and choose the branch to deploy.
4. Use these settings:

   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --workers 2 --bind 0.0.0.0:$PORT app:app`
   - **Instance type:** `Free` for a demo, or a paid instance for production use

5. Click **Create Web Service**. Render installs the dependencies, starts Gunicorn, and provides a public `.onrender.com` URL.

The application reads Render's `PORT` environment variable automatically. The `trained_model.pkl` file is loaded from the project directory, so it must be present in the repository or supplied through a separate model-storage workflow.

## Updating the model

Run the training script locally:

```powershell
python train_model.py
```

Commit and push the updated `trained_model.pkl`. Render will redeploy automatically if auto-deploy is enabled.

## Project files

- `app.py` - Flask application and server entry point
- `trained_model.pkl` - saved preprocessing pipeline and classifier
- `templates/index.html` - candidate evaluation page
- `static/styles.css` - responsive web styling
- `train_model.py` - model training and evaluation
- `requirements.txt` - Python dependencies