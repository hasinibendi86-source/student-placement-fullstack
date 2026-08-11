# 🎓 Student Placement Prediction System (Full-Stack)

A beginner-friendly full-stack Machine Learning project that predicts
whether a student is likely to be **Placed** or **Not Placed**, using a
plain HTML/CSS/JavaScript frontend, a Flask REST API backend, and a
Scikit-learn Logistic Regression model.

---

## 1. Problem Statement

Colleges want to identify, ahead of placement season, which students are
likely to be placed based on measurable data — academics, skills, and
internship experience — so they can offer extra guidance to students who
need it. Doing this manually is slow and inconsistent. This project
automates the estimate with a trained Machine Learning model, served
through a REST API and a simple website.

## 2. Objective

- Train a Logistic Regression model on student placement data.
- Serve predictions through a Flask REST API (`POST /predict`).
- Build a separate, static HTML/CSS/JS frontend that calls that API using
  the Fetch API and displays the result.
- Keep the whole stack simple enough for a beginner to build, run, and
  explain in a viva.

## 3. Features

- Clean, distinct "academic record" themed UI (not a generic template)
- Full client-side validation (range checks) + server-side validation (security)
- Fetch API call from the browser to the Flask backend
- JSON REST API with clear error messages for bad input
- Displays PLACED / NOT PLACED with a probability percentage and progress meter
- Loading state on the Predict button, error banner for network/server failures
- Reset button to clear the form
- Fully responsive, mobile-friendly layout

## 4. Technology Stack

| Layer        | Technology                              |
|--------------|------------------------------------------|
| Frontend     | HTML5, CSS3, vanilla JavaScript, Fetch API |
| Backend      | Python, Flask, Flask-CORS                |
| Machine Learning | Scikit-learn (Logistic Regression), Pandas, NumPy, Joblib |
| Dataset      | CSV file (`student_placement.csv`)       |

No React/Angular/Vue, no Streamlit, no database — deliberately kept simple.

## 5. Project Architecture

```
 ┌─────────────────────────┐         POST /predict          ┌──────────────────────────┐
 │        FRONTEND          │  ─────────────────────────▶   │         BACKEND           │
 │  index.html + style.css  │        (JSON payload)          │   Flask REST API (app.py) │
 │        + script.js       │                                 │                            │
 │                          │  ◀─────────────────────────    │  1. Validate input        │
 │  - Collects form input   │      JSON response              │  2. Load placement_model  │
 │  - Validates in-browser  │   {prediction, probability}     │     .pkl (joblib)         │
 │  - Calls fetch()         │                                 │  3. Scale + predict       │
 │  - Renders result card   │                                 │  4. Return JSON           │
 └─────────────────────────┘                                 └──────────────┬─────────────┘
                                                                              │
                                                                              │ loads
                                                                              ▼
                                                              ┌──────────────────────────┐
                                                              │   placement_model.pkl     │
                                                              │  (LogisticRegression +    │
                                                              │   StandardScaler, saved   │
                                                              │   by train_model.py)      │
                                                              └──────────────┬─────────────┘
                                                                              │ trained from
                                                                              ▼
                                                              ┌──────────────────────────┐
                                                              │  student_placement.csv     │
                                                              │  (300 sample records)      │
                                                              └──────────────────────────┘
```

### Data flow, step by step

1. The user fills the form in the browser (`index.html`) and clicks **Predict Placement**.
2. `script.js` validates every field in the browser first (fast feedback, no network needed).
3. `script.js` sends the form values as JSON to the Flask API using `fetch("http://127.0.0.1:5000/predict", { method: "POST", ... })`.
4. Flask (`app.py`) receives the request, re-validates the data (never trust the client alone), and loads the already-trained model from `placement_model.pkl`.
5. The model scales the input (using the same `StandardScaler` fitted during training) and calls `model.predict()` / `model.predict_proba()`.
6. Flask returns a JSON response like `{"prediction": "Placed", "probability": 87.5}`.
7. `script.js` receives the JSON and updates the result card: the PLACED/NOT PLACED stamp, the probability bar, and a summary of what was submitted.

## 6. Folder Structure

```text
student-placement-fullstack/
│
├── frontend/
│   ├── index.html          # Placement prediction form + result card
│   ├── style.css            # Styling (responsive, "academic record" theme)
│   └── script.js             # Validation + Fetch API calls to the backend
│
├── backend/
│   ├── app.py                # Flask REST API (POST /predict, GET /health)
│   ├── train_model.py        # Trains and saves the Logistic Regression model
│   ├── student_placement.csv # Sample dataset (300 records)
│   └── placement_model.pkl   # Saved trained model + scaler (created by train_model.py)
│
├── requirements.txt
└── README.md
```

## 7. Dataset Description

File: `backend/student_placement.csv` — 300 sample student records.

| Column               | Description                               |
|-----------------------|--------------------------------------------|
| CGPA                 | Overall CGPA (0–10)                        |
| 10th_Percentage      | 10th grade marks (%)                        |
| 12th_Percentage      | 12th grade marks (%)                        |
| Technical_Score      | Technical skill score (0–100)               |
| Aptitude_Score       | Aptitude test score (0–100)                 |
| Communication_Score  | Communication skill score (0–100)           |
| Internship           | 0 = No internship, 1 = Internship           |
| Placement (target)   | 0 = Not Placed, 1 = Placed                  |

The dataset was generated with realistic relationships (higher CGPA, skill
scores, and internship experience increase placement likelihood), plus
random noise, so the model has meaningful — but not perfectly obvious —
patterns to learn.

## 8. How the Machine Learning Model Works

1. `train_model.py` loads the CSV and splits it into features (CGPA,
   percentages, scores, internship) and the target (`Placement`).
2. The data is split 80/20 into training and testing sets.
3. `StandardScaler` scales all features to a comparable range (important
   for Logistic Regression, since CGPA is on a 0–10 scale but scores are 0–100).
4. `LogisticRegression` is trained on the scaled training data.
5. The model is evaluated on the unseen 20% test set — this project
   achieves **~90% accuracy**.
6. The trained model **and** the fitted scaler are bundled together and
   saved with `joblib.dump()` into `placement_model.pkl`, so the backend
   can apply the exact same scaling at prediction time.

**Why Logistic Regression?** This is a binary classification problem
(Placed / Not Placed). Logistic Regression is simple, fast, and — crucially
— outputs a *probability*, not just a label, which is exactly what the UI
needs to show a confidence percentage.

## 9. Backend API Reference

### `POST /predict`

Request body (JSON):
```json
{
  "cgpa": 8.2,
  "tenth": 85,
  "twelfth": 88,
  "technical": 80,
  "aptitude": 75,
  "communication": 82,
  "internship": 1
}
```

Success response (200):
```json
{
  "prediction": "Placed",
  "probability": 87.5
}
```

Validation error response (400):
```json
{ "error": "CGPA must be between 0 and 10." }
```

Server error response (500):
```json
{ "error": "Model not found on server. Please run train_model.py first." }
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

## 10. Installation Steps (Windows / VS Code / Command Prompt)

1. Install **Python 3.8+** from [python.org](https://www.python.org/) if you don't have it.
2. Open Command Prompt (or the VS Code terminal) in the project's root folder:
   ```bash
   cd student-placement-fullstack
   ```
3. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 11. How to Train the Model

```bash
cd backend
python train_model.py
```

This reads `student_placement.csv`, trains the model, prints the accuracy,
and saves `placement_model.pkl` in the `backend/` folder. You should see:
```
Model Accuracy: 90.00%
Saved trained model to 'placement_model.pkl'
```

## 12. How to Start the Flask Backend

Still inside the `backend/` folder:
```bash
python app.py
```
You should see:
```
Model loaded successfully from placement_model.pkl
 * Running on http://127.0.0.1:5000
```
Leave this terminal window open — the server needs to keep running while
you use the frontend.

## 13. How to Open the Frontend

The frontend is a set of static files, so you have two options:

**Option A — Simplest (double-click):**
Open `frontend/index.html` directly in your browser (double-click the file,
or right-click → Open with → your browser). Most browsers allow `fetch()`
calls from a local file to `http://127.0.0.1:5000` without issues, since
the backend has CORS enabled.

**Option B — Recommended (serve it locally):**
Some browsers restrict requests from `file://` pages. To avoid any issues,
serve the frontend with Python's built-in server, in a **second** terminal
window (keep the Flask server running in the first one):
```bash
cd frontend
python -m http.server 5500
```
Then open your browser to: `http://127.0.0.1:5500`

Either way, make sure the Flask backend (Step 12) is running first, since
the frontend calls `http://127.0.0.1:5000/predict`.

## 14. Example Input and Output

**Example Input:**
| Field                | Value |
|-----------------------|-------|
| CGPA                  | 8.2   |
| 10th Percentage       | 85    |
| 12th Percentage       | 88    |
| Technical Score       | 80    |
| Aptitude Score        | 75    |
| Communication Score   | 82    |
| Internship             | Yes   |

**Example Output (shown on the result card):**
```
Placement Prediction
Status: PLACED
Confidence: 96.4%
```

**Example of a weaker profile:**
| CGPA | 5.5 | 10th | 55 | 12th | 52 | Technical | 35 | Aptitude | 30 | Communication | 40 | Internship | No |

```
Placement Prediction
Status: NOT PLACED
Confidence: 2.1%
```
(Here "Confidence" always refers to the probability of the "Placed" class;
a low number means the model is confident the student will **not** be placed.)

## 15. Future Enhancements

- Train on a larger, real-world placement dataset.
- Add more features (backlogs, certifications, extracurriculars).
- Try other models (Random Forest, XGBoost) and compare accuracy.
- Deploy the backend (Render/Railway) and frontend (Netlify/Vercel) so it's live on the internet.
- Add a chart showing which factors most influenced the prediction.
- Add user accounts and store prediction history in a database.

---

## 16. Viva Questions & Answers

**Q1. What problem does this project solve?**
It predicts whether a student is likely to be placed, using their
academic and skill data, so colleges can identify students who may need
extra support before placement season.

**Q2. Why did you use Logistic Regression instead of another algorithm?**
It's a binary classification problem (Placed vs Not Placed). Logistic
Regression is simple, fast, easy to explain, and naturally outputs a
probability — which we use to show the confidence percentage in the UI.

**Q3. Why do you scale the features before training?**
CGPA is on a 0–10 scale, while scores are on a 0–100 scale. Without
scaling, Logistic Regression would treat larger-scale features (like
scores) as automatically more important, purely because of their size.
`StandardScaler` puts every feature on the same footing.

**Q4. Why save both the model and the scaler in the same `.pkl` file?**
At prediction time, new input must be scaled in exactly the same way the
training data was. Saving them together guarantees the backend always
uses the correct, matching scaler — never a mismatched one.

**Q5. Why is the frontend separate from the backend?**
It mirrors how real-world web apps are built: the frontend (what the user
sees) and the backend (business logic + ML model) are independent
services that communicate over HTTP. This also means the backend could
serve a mobile app or another client in the future without any changes.

**Q6. What is CORS and why is it needed here?**
CORS (Cross-Origin Resource Sharing) is a browser security feature that
blocks a webpage from calling an API on a different origin (domain/port)
unless that API explicitly allows it. Since the frontend and Flask
backend run on different ports, `flask-cors` is used to allow the
frontend to call the API.

**Q7. What does the Fetch API do in this project?**
`fetch()` is used in `script.js` to send the form data as JSON to the
Flask `/predict` endpoint and receive the prediction back, without
reloading the page.

**Q8. How is the input validated?**
Twice: first in the browser (`script.js`) for instant feedback, and again
in Flask (`app.py`) before prediction — because client-side validation
alone can be bypassed, so the server must never trust it blindly.

**Q9. What does "probability" mean in the output?**
It's the model's estimated probability (0–100%) that the student belongs
to the "Placed" class, calculated using `model.predict_proba()`. Above
50% probability, the model predicts "Placed"; otherwise, "Not Placed."

**Q10. What is the model's accuracy, and what does it mean?**
Around 90% on the test set (the 20% of data the model never saw during
training). This means the model correctly predicted the outcome for 9
out of 10 students in that held-out set.

**Q11. How would you improve this project further?**
Use a real dataset, try more advanced algorithms, add more features, and
deploy both frontend and backend online — all listed in the Future
Enhancements section above.
