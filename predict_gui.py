"""
predict_gui.py
----------------
A simple Tkinter desktop GUI for the ML Based Admission Selection System.
Loads trained_model.pkl (produced by train_model.py) and lets a registration
officer type in a candidate's details to get an instant Admit/Reject
recommendation with a confidence score.

Run with:  python predict_gui.py
"""

import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import pandas as pd

MODEL_PATH = "trained_model.pkl"


class AdmissionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Based Admission Selection System")
        self.root.geometry("480x560")
        self.root.resizable(False, False)

        try:
            bundle = joblib.load(MODEL_PATH)
            self.pipeline = bundle["pipeline"]
            self.model_name = bundle["model_name"]
        except FileNotFoundError:
            messagebox.showerror(
                "Model not found",
                f"Could not find {MODEL_PATH}.\nRun train_model.py first.",
            )
            self.pipeline = None
            self.model_name = "N/A"

        self._build_ui()

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Admission Candidate Evaluation",
            font=("Helvetica", 16, "bold"),
        )
        title.pack(pady=(15, 2))

        subtitle = tk.Label(
            self.root, text=f"Model in use: {self.model_name}", font=("Helvetica", 9, "italic")
        )
        subtitle.pack(pady=(0, 15))

        form = tk.Frame(self.root)
        form.pack(padx=20, fill="x")

        self.entries = {}

        def add_numeric_field(label, key, default=""):
            row = tk.Frame(form)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, width=26, anchor="w").pack(side="left")
            entry = tk.Entry(row)
            entry.insert(0, default)
            entry.pack(side="right", expand=True, fill="x")
            self.entries[key] = entry

        def add_dropdown(label, key, options):
            row = tk.Frame(form)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, width=26, anchor="w").pack(side="left")
            var = tk.StringVar(value=options[0])
            dropdown = ttk.Combobox(row, textvariable=var, values=options, state="readonly")
            dropdown.pack(side="right", expand=True, fill="x")
            self.entries[key] = var

        add_numeric_field("Age:", "age", "18")
        add_numeric_field("UTME Score (0-400):", "utme_score", "250")
        add_numeric_field("Post-UTME Score (0-100):", "post_utme_score", "60")
        add_numeric_field("O'level Points (0-45):", "olevel_points", "35")
        add_numeric_field("Subject Relevance (0-1):", "subject_relevance", "0.8")
        add_numeric_field("Interview Score (0-100):", "interview_score", "65")
        add_dropdown("Catchment Category:", "catchment_category",
                     ["Catchment", "Non-Catchment", "ELDS"])
        add_dropdown("Course Choice:", "course_choice",
                     ["First Choice", "Second Choice"])

        predict_btn = tk.Button(
            self.root, text="Evaluate Candidate", command=self.predict,
            bg="#2e7d32", fg="white", font=("Helvetica", 11, "bold"), pady=8
        )
        predict_btn.pack(pady=20, fill="x", padx=20)

        self.result_label = tk.Label(
            self.root, text="", font=("Helvetica", 13, "bold"), wraplength=420, justify="center"
        )
        self.result_label.pack(pady=10)

    def predict(self):
        if self.pipeline is None:
            messagebox.showerror("Error", "No trained model loaded.")
            return

        try:
            candidate = {
                "age": float(self.entries["age"].get()),
                "utme_score": float(self.entries["utme_score"].get()),
                "post_utme_score": float(self.entries["post_utme_score"].get()),
                "olevel_points": float(self.entries["olevel_points"].get()),
                "subject_relevance": float(self.entries["subject_relevance"].get()),
                "interview_score": float(self.entries["interview_score"].get()),
                "catchment_category": self.entries["catchment_category"].get(),
                "course_choice": self.entries["course_choice"].get(),
            }
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numbers for all numeric fields.")
            return

        X_new = pd.DataFrame([candidate])
        prediction = self.pipeline.predict(X_new)[0]
        probability = self.pipeline.predict_proba(X_new)[0][1]

        if prediction == 1:
            self.result_label.config(
                text=f"✅ ADMITTED\nConfidence: {probability * 100:.1f}%",
                fg="#2e7d32",
            )
        else:
            self.result_label.config(
                text=f"❌ NOT ADMITTED\nAdmission likelihood: {probability * 100:.1f}%",
                fg="#c62828",
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = AdmissionApp(root)
    root.mainloop()
