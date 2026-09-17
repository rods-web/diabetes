from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import joblib, numpy as np, json, os, csv, requests
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change_me_secret_2025")

RECAPTCHA_SITE_KEY = os.environ.get("RECAPTCHA_SITE_KEY", "6LeJxcAtAAAAAFyvZv82q5Fj5PI1rHHcPOl2IIAl")
RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY", "6LeJxcAtAAAAAEcwhg26w42KzWQntotyVBkM-_j6")

USERS_FILE = "users.json"
HISTORY_FILE = "history.json"
FEEDBACK_FILE = "feedback.json"
MODELS_DIR = "models"

def load_json(path, default):
    if not os.path.exists(path): return default
    with open(path, "r") as f: return json.load(f)

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

def load_users(): return load_json(USERS_FILE, {})
def save_users(u): save_json(USERS_FILE, u)
def load_history(): return load_json(HISTORY_FILE, {})
def save_history(h): save_json(HISTORY_FILE, h)

def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if "user" not in session:
            flash("Please log in first.", "error")
            return redirect(url_for("login"))
        return f(*a, **k)
    return w

def verify_recaptcha(token):
    if not token: return False
    try:
        r = requests.post("https://www.google.com/recaptcha/api/siteverify",
                          data={"secret": RECAPTCHA_SECRET_KEY, "response": token}, timeout=5)
        return r.json().get("success", False)
    except Exception:
        return False

model = joblib.load(f"{MODELS_DIR}/enhanced_rf.pkl")
scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
feature_mask = joblib.load(f"{MODELS_DIR}/feature_mask.pkl")
feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")

comparison_data = []
if os.path.exists(f"{MODELS_DIR}/comparison_results.csv"):
    with open(f"{MODELS_DIR}/comparison_results.csv") as f:
        for row in csv.DictReader(f):
            comparison_data.append({
                "model": row["Model"],
                "accuracy": float(row["Accuracy"]),
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1"]),
                "auc": float(row["AUC"]),
            })

CHATBOT_RESPONSES = [
    {"k": ["what", "website", "site", "purpose", "about"], "r": "DiaPredict is a web-based diabetes risk prediction system using Enhanced Random Forest. Note: Not a medical diagnosis."},
    {"k": ["diabetes", "what is", "define"], "r": "Diabetes mellitus is a chronic disorder where blood sugar is too high. Types: Type 1, Type 2, gestational."},
    {"k": ["symptom", "signs", "feel"], "r": "Symptoms: frequent urination, excessive thirst, weight loss, fatigue, blurred vision."},
    {"k": ["risk", "factor", "cause"], "r": "Risk factors: age 45+, family history, obesity, high BP, inactivity, poor diet."},
    {"k": ["prevent", "avoid", "reduce"], "r": "Prevention: healthy weight, exercise, fiber, less sugar, no smoking."},
    {"k": ["treatment", "cure", "medicine", "insulin"], "r": "Managed but not cured. Consult a doctor."},
    {"k": ["type", "gestational"], "r": "Type 1: autoimmune. Type 2: insulin resistance. Gestational: pregnancy."},
    {"k": ["accuracy", "model", "random forest"], "r": "Enhanced Random Forest (HFS-RF) optimized with feature selection, SMOTE, hyperparameter tuning. See dashboard."},
    {"k": ["how", "use", "predict", "start"], "r": "Steps: 1) Login, 2) New Prediction, 3) Enter values, 4) Generate Prediction."},
    {"k": ["history", "past", "records"], "r": "View your prediction history in the sidebar."},
    {"k": ["privacy", "data", "secure", "safe"], "r": "Data stored locally. Passwords hashed. Only you can see your predictions."},
    {"k": ["doctor", "medical", "advice"], "r": "Not a medical diagnosis. Consult a licensed physician."},
    {"k": ["hello", "hi", "hey"], "r": "Hi! Ask me about diabetes, this website, or how to use predictions."},
    {"k": ["thank", "thanks"], "r": "You're welcome!"},
    {"k": ["help"], "r": "I can answer about: this website, diabetes basics, how to predict, prevention, privacy."},
]
DEFAULT_RESPONSE = "I'm not sure. Try asking: 'What is diabetes?' or 'How do I predict?'"

def get_bot_response(msg):
    m = msg.lower()
    best, score = None, 0
    for e in CHATBOT_RESPONSES:
        s = sum(1 for kw in e["k"] if kw in m)
        if s > score: score, best = s, e["r"]
    return best or DEFAULT_RESPONSE

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    d = request.get_json() or {}
    m = d.get("message", "").strip()
    if not m: return jsonify({"reply": "Please type a message."})
    return jsonify({"reply": get_bot_response(m)})

@app.route("/feedback", methods=["GET", "POST"])
@login_required
def feedback():
    if request.method == "POST":
        r = {k: int(request.form.get(k, 0)) for k in ["accuracy", "reliability", "efficiency", "usability", "overall"]}
        if any(v < 1 or v > 5 for v in r.values()):
            flash("Rate all items 1-5.", "error")
            return redirect(url_for("feedback"))
        fb = load_json(FEEDBACK_FILE, [])
        fb.append({"user": session["user"], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "ratings": r, "comment": request.form.get("comment", "").strip()})
        save_json(FEEDBACK_FILE, fb)
        flash("Thanks for your feedback!", "success")
        return redirect(url_for("feedback_results"))
    return render_template("feedback.html", user=session["user"])

@app.route("/feedback/results")
@login_required
def feedback_results():
    fb = load_json(FEEDBACK_FILE, [])
    if not fb:
        return render_template("feedback_results.html", user=session["user"], total=0, means={}, overall_mean=0, interpretation="No data yet")
    keys = ["accuracy", "reliability", "efficiency", "usability", "overall"]
    means = {k: round(sum(f["ratings"][k] for f in fb) / len(fb), 2) for k in keys}
    om = round(sum(means.values()) / len(means), 2)
    interp = "Excellent" if om >= 4.5 else "Very Good" if om >= 3.5 else "Good" if om >= 2.5 else "Fair" if om >= 1.5 else "Poor"
    return render_template("feedback_results.html", user=session["user"], total=len(fb), means=means, overall_mean=om, interpretation=interp)

@app.route("/")
def landing():
    if "user" in session: return redirect(url_for("dashboard"))
    return render_template("landing.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        if not verify_recaptcha(request.form.get("g-recaptcha-response")):
            flash("Please complete the CAPTCHA.", "error")
            return redirect(url_for("signup"))
        u = request.form.get("username", "").strip()
        e = request.form.get("email", "").strip()
        p = request.form.get("password", "")
        if not u or not e or not p:
            flash("All fields required.", "error"); return redirect(url_for("signup"))
        users = load_users()
        if u in users:
            flash("Username exists.", "error"); return redirect(url_for("signup"))
        users[u] = {"email": e, "password": generate_password_hash(p), "created": datetime.now().isoformat()}
        save_users(users)
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html", site_key=RECAPTCHA_SITE_KEY)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if not verify_recaptcha(request.form.get("g-recaptcha-response")):
            flash("Please complete the CAPTCHA.", "error")
            return redirect(url_for("login"))
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")
        users = load_users()
        if u not in users or not check_password_hash(users[u]["password"], p):
            flash("Invalid credentials.", "error"); return redirect(url_for("login"))
        session["user"] = u
        session["email"] = users[u]["email"]
        flash(f"Welcome back, {u}!", "success")
        return redirect(url_for("dashboard"))
    return render_template("login.html", site_key=RECAPTCHA_SITE_KEY)

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("landing"))

@app.route("/dashboard")
@login_required
def dashboard():
    h = load_history().get(session["user"], [])
    return render_template("dashboard.html", user=session["user"], email=session.get("email"),
        comparison=comparison_data,
        total_predictions=len(h),
        high_risk_count=sum(1 for x in h if x["prediction"] == "High Risk"),
        low_risk_count=sum(1 for x in h if x["prediction"] == "Low Risk"))

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            inp = {f: float(request.form[f]) for f in feature_names}
            arr = np.array([inp[f] for f in feature_names]).reshape(1, -1)
            a_s = scaler.transform(arr)[:, feature_mask]
            pred = model.predict(a_s)[0]
            proba = model.predict_proba(a_s)[0][1]
            rl = "High Risk" if pred == 1 else "Low Risk"
            conf = round(proba * 100, 2)
            h = load_history()
            h.setdefault(session["user"], []).append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": rl, "confidence": conf, "inputs": inp})
            save_history(h)
            return render_template("result.html", user=session["user"], prediction=rl, confidence=conf)
        except Exception as e:
            flash(f"Error: {e}", "error"); return redirect(url_for("predict"))
    return render_template("predict.html", user=session["user"], features=feature_names)

@app.route("/history")
@login_required
def history():
    uh = load_history().get(session["user"], [])
    total = len(uh)
    hr = sum(1 for x in uh if x["prediction"] == "High Risk")
    return render_template("history.html", user=session["user"], history=list(reversed(uh)),
        total=total, high_risk=hr, low_risk=total - hr,
        avg_conf=round(sum(x["confidence"] for x in uh) / total, 2) if total else 0,
        feature_names=feature_names)

@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    h = load_history()
    h[session["user"]] = []
    save_history(h)
    flash("History cleared.", "success")
    return redirect(url_for("history"))

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    users = load_users()
    u = session["user"]
    if request.method == "POST":
        a = request.form.get("action")
        if a == "update_email":
            ne = request.form.get("email", "").strip()
            if ne:
                users[u]["email"] = ne; save_users(users); session["email"] = ne
                flash("Email updated.", "success")
        elif a == "change_password":
            cur, nw, cf = (request.form.get(k, "") for k in ["current_password", "new_password", "confirm_password"])
            if not check_password_hash(users[u]["password"], cur):
                flash("Current password wrong.", "error")
            elif len(nw) < 6:
                flash("Password must be 6+ chars.", "error")
            elif nw != cf:
                flash("Passwords don't match.", "error")
            else:
                users[u]["password"] = generate_password_hash(nw); save_users(users)
                flash("Password changed.", "success")
        elif a == "delete_account":
            if request.form.get("confirm_delete") != "DELETE":
                flash("Type DELETE to confirm.", "error")
            else:
                del users[u]; save_users(users)
                h = load_history(); h.pop(u, None); save_history(h)
                session.clear()
                flash("Account deleted.", "success")
                return redirect(url_for("landing"))
        return redirect(url_for("settings"))
    return render_template("settings.html", user=u, email=users[u].get("email", ""), created=users[u].get("created", ""))

if __name__ == "__main__":
    app.run(host=os.environ.get("FLASK_HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", 5000)), debug=False)
