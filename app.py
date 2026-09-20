from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import joblib, numpy as np, json, os, csv, requests

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change_me_secret_2025")

RECAPTCHA_SITE_KEY = os.environ.get("RECAPTCHA_SITE_KEY", "6LeJxcAtAAAAAFyvZv82q5Fj5PI1rHHcPOl2IIAl")
RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY", "6LeJxcAtAAAAAEcwhg26w42KzWQntotyVBkM-_j6")

# ============================================================
# SQLITE DATABASE
# ============================================================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

MODELS_DIR = "models"


# ============================================================
# DATABASE MODELS
# ============================================================
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created = db.Column(db.DateTime, default=datetime.utcnow)


class LoginLog(db.Model):
    __tablename__ = "login_logs"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    action = db.Column(db.String(20))
    ip = db.Column(db.String(50))
    user_agent = db.Column(db.String(255))


class Prediction(db.Model):
    __tablename__ = "history"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), db.ForeignKey("users.username"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    inputs_json = db.Column(db.Text, nullable=False)


class Feedback(db.Model):
    __tablename__ = "feedback"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    accuracy = db.Column(db.Integer)
    reliability = db.Column(db.Integer)
    efficiency = db.Column(db.Integer)
    usability = db.Column(db.Integer)
    overall = db.Column(db.Integer)
    comment = db.Column(db.Text)


with app.app_context():
    db.create_all()
    if not User.query.filter_by(username="admin").first():
        admin = User(
            username="admin",
            email="admin@diapredict.com",
            password_hash=generate_password_hash("admin123"),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()
        print("✅ Default admin created — username: admin, password: admin123")


# ============================================================
# DECORATORS
# ============================================================
def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if "user" not in session:
            flash("Please log in first.", "error")
            return redirect(url_for("login"))
        return f(*a, **k)
    return w


def admin_required(f):
    @wraps(f)
    def w(*a, **k):
        if "user" not in session or not session.get("is_admin"):
            flash("Admin access required.", "error")
            return redirect(url_for("dashboard"))
        return f(*a, **k)
    return w


def verify_recaptcha(token):
    if not token:
        return False
    try:
        r = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": RECAPTCHA_SECRET_KEY, "response": token},
            timeout=5
        )
        return r.json().get("success", False)
    except Exception:
        return False


# ============================================================
# LOAD ML MODEL
# ============================================================
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


# ============================================================
# CHATBOT
# ============================================================
CHATBOT_RESPONSES = [
    {"k": ["what", "website", "site", "purpose", "about"], "r": "DiaPredict is a web-based diabetes risk prediction system using Enhanced Random Forest."},
    {"k": ["diabetes", "what is", "define"], "r": "Diabetes mellitus is a chronic disorder where blood sugar is too high."},
    {"k": ["symptom", "signs", "feel"], "r": "Symptoms: frequent urination, excessive thirst, weight loss, fatigue."},
    {"k": ["risk", "factor", "cause"], "r": "Risk factors: age 45+, family history, obesity, high BP, inactivity."},
    {"k": ["prevent", "avoid", "reduce"], "r": "Prevention: healthy weight, exercise, fiber, less sugar."},
    {"k": ["treatment", "cure", "medicine", "insulin"], "r": "Managed but not cured. Consult a doctor."},
    {"k": ["type", "gestational"], "r": "Type 1, Type 2, and gestational diabetes."},
    {"k": ["accuracy", "model", "random forest"], "r": "Enhanced Random Forest with feature selection, SMOTE, tuning."},
    {"k": ["how", "use", "predict", "start"], "r": "1) Login, 2) New Prediction, 3) Enter values, 4) Submit."},
    {"k": ["history", "past", "records"], "r": "Check the sidebar for your prediction history."},
    {"k": ["privacy", "data", "secure"], "r": "Passwords hashed. Only you can see your data."},
    {"k": ["doctor", "medical"], "r": "Not a medical diagnosis. Consult a licensed physician."},
    {"k": ["hello", "hi", "hey"], "r": "Hi! Ask me about diabetes or this website."},
    {"k": ["thank", "thanks"], "r": "You're welcome!"},
    {"k": ["help"], "r": "Ask about: this website, diabetes basics, predictions, privacy."},
]
DEFAULT_RESPONSE = "Try asking: 'What is diabetes?' or 'How do I predict?'"


def get_bot_response(msg):
    m = msg.lower()
    best, score = None, 0
    for e in CHATBOT_RESPONSES:
        s = sum(1 for kw in e["k"] if kw in m)
        if s > score:
            score, best = s, e["r"]
    return best or DEFAULT_RESPONSE


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    d = request.get_json() or {}
    m = d.get("message", "").strip()
    if not m:
        return jsonify({"reply": "Please type a message."})
    return jsonify({"reply": get_bot_response(m)})


# ============================================================
# FEEDBACK
# ============================================================
@app.route("/feedback", methods=["GET", "POST"])
@login_required
def feedback():
    if request.method == "POST":
        fb = Feedback(
            username=session["user"],
            accuracy=int(request.form.get("accuracy", 0)),
            reliability=int(request.form.get("reliability", 0)),
            efficiency=int(request.form.get("efficiency", 0)),
            usability=int(request.form.get("usability", 0)),
            overall=int(request.form.get("overall", 0)),
            comment=request.form.get("comment", "").strip()
        )
        if any(v < 1 or v > 5 for v in [fb.accuracy, fb.reliability, fb.efficiency, fb.usability, fb.overall]):
            flash("Rate all items 1-5.", "error")
            return redirect(url_for("feedback"))
        db.session.add(fb)
        db.session.commit()
        flash("Thanks for your feedback!", "success")
        return redirect(url_for("feedback_results"))
    return render_template("feedback.html", user=session["user"])


@app.route("/feedback/results")
@login_required
def feedback_results():
    fb = Feedback.query.all()
    if not fb:
        return render_template("feedback_results.html", user=session["user"], total=0, means={}, overall_mean=0, interpretation="No data yet")
    keys = ["accuracy", "reliability", "efficiency", "usability", "overall"]
    means = {k: round(sum(getattr(f, k) for f in fb) / len(fb), 2) for k in keys}
    om = round(sum(means.values()) / len(means), 2)
    interp = "Excellent" if om >= 4.5 else "Very Good" if om >= 3.5 else "Good" if om >= 2.5 else "Fair" if om >= 1.5 else "Poor"
    return render_template("feedback_results.html", user=session["user"], total=len(fb), means=means, overall_mean=om, interpretation=interp)


# ============================================================
# PUBLIC ROUTES
# ============================================================
@app.route("/")
def landing():
    if "user" in session:
        if session.get("is_admin"):
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("dashboard"))
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
            flash("All fields required.", "error")
            return redirect(url_for("signup"))
        if User.query.filter_by(username=u).first():
            flash("Username exists.", "error")
            return redirect(url_for("signup"))
        user = User(username=u, email=e, password_hash=generate_password_hash(p))
        db.session.add(user)
        db.session.commit()
        db.session.add(LoginLog(username=u, action="signup", ip=request.remote_addr,
                                user_agent=request.headers.get("User-Agent", "")[:255]))
        db.session.commit()
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
        user = User.query.filter_by(username=u).first()
        if not user or not check_password_hash(user.password_hash, p):
            flash("Invalid credentials.", "error")
            return redirect(url_for("login"))
        session["user"] = user.username
        session["email"] = user.email
        session["is_admin"] = user.is_admin
        db.session.add(LoginLog(username=u, action="login", ip=request.remote_addr,
                                user_agent=request.headers.get("User-Agent", "")[:255]))
        db.session.commit()
        flash(f"Welcome back, {u}!", "success")
        if user.is_admin:
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("dashboard"))
    return render_template("login.html", site_key=RECAPTCHA_SITE_KEY)


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("landing"))


# ============================================================
# PROTECTED ROUTES
# ============================================================
@app.route("/dashboard")
@login_required
def dashboard():
    preds = Prediction.query.filter_by(username=session["user"]).all()
    return render_template("dashboard.html",
        user=session["user"], email=session.get("email"),
        comparison=comparison_data,
        total_predictions=len(preds),
        high_risk_count=sum(1 for p in preds if p.prediction == "High Risk"),
        low_risk_count=sum(1 for p in preds if p.prediction == "Low Risk"))


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
            record = Prediction(username=session["user"], prediction=rl,
                                confidence=conf, inputs_json=json.dumps(inp))
            db.session.add(record)
            db.session.commit()
            return render_template("result.html", user=session["user"], prediction=rl, confidence=conf)
        except Exception as e:
            flash(f"Error: {e}", "error")
            return redirect(url_for("predict"))
    return render_template("predict.html", user=session["user"], features=feature_names)


@app.route("/history")
@login_required
def history():
    preds = Prediction.query.filter_by(username=session["user"]).order_by(Prediction.timestamp.desc()).all()
    total = len(preds)
    hr = sum(1 for p in preds if p.prediction == "High Risk")
    history_data = [{
        "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": p.prediction,
        "confidence": p.confidence,
        "inputs": json.loads(p.inputs_json)
    } for p in preds]
    return render_template("history.html",
        user=session["user"], history=history_data,
        total=total, high_risk=hr, low_risk=total - hr,
        avg_conf=round(sum(p.confidence for p in preds) / total, 2) if total else 0,
        feature_names=feature_names)


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    Prediction.query.filter_by(username=session["user"]).delete()
    db.session.commit()
    flash("History cleared.", "success")
    return redirect(url_for("history"))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    user = User.query.filter_by(username=session["user"]).first()
    if request.method == "POST":
        a = request.form.get("action")
        if a == "update_email":
            ne = request.form.get("email", "").strip()
            if ne:
                user.email = ne
                db.session.commit()
                session["email"] = ne
                flash("Email updated.", "success")
        elif a == "change_password":
            cur, nw, cf = (request.form.get(k, "") for k in ["current_password", "new_password", "confirm_password"])
            if not check_password_hash(user.password_hash, cur):
                flash("Current password wrong.", "error")
            elif len(nw) < 6:
                flash("Password must be 6+ chars.", "error")
            elif nw != cf:
                flash("Passwords don't match.", "error")
            else:
                user.password_hash = generate_password_hash(nw)
                db.session.commit()
                flash("Password changed.", "success")
        elif a == "delete_account":
            if request.form.get("confirm_delete") != "DELETE":
                flash("Type DELETE to confirm.", "error")
            else:
                Prediction.query.filter_by(username=user.username).delete()
                LoginLog.query.filter_by(username=user.username).delete()
                Feedback.query.filter_by(username=user.username).delete()
                db.session.delete(user)
                db.session.commit()
                session.clear()
                flash("Account deleted.", "success")
                return redirect(url_for("landing"))
        return redirect(url_for("settings"))
    return render_template("settings.html", user=user.username, email=user.email,
                           created=user.created.isoformat() if user.created else "")


# ============================================================
# ADMIN ROUTES
# ============================================================
@app.route("/admin")
@admin_required
def admin_dashboard():
    users = User.query.order_by(User.created.desc()).all()
    logins = LoginLog.query.order_by(LoginLog.timestamp.desc()).limit(50).all()
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(50).all()

    # -------- Feedback aggregation --------
    all_fb = Feedback.query.all()
    feedback_means = {}
    overall_mean = 0
    interpretation = "No data yet"

    if all_fb:
        keys = ["accuracy", "reliability", "efficiency", "usability", "overall"]
        feedback_means = {
            k: round(sum(getattr(f, k) for f in all_fb) / len(all_fb), 2)
            for k in keys
        }
        overall_mean = round(sum(feedback_means.values()) / len(feedback_means), 2)
        if overall_mean >= 4.5:
            interpretation = "Excellent"
        elif overall_mean >= 3.5:
            interpretation = "Very Good"
        elif overall_mean >= 2.5:
            interpretation = "Good"
        elif overall_mean >= 1.5:
            interpretation = "Fair"
        else:
            interpretation = "Poor"

    # -------- Risk distribution --------
    all_preds = Prediction.query.all()
    high_risk_count = sum(1 for p in all_preds if p.prediction == "High Risk")
    low_risk_count = sum(1 for p in all_preds if p.prediction == "Low Risk")

    # -------- Top active users --------
    user_counts = {}
    for p in all_preds:
        user_counts[p.username] = user_counts.get(p.username, 0) + 1

    top_users_list = []
    for uname, count in sorted(user_counts.items(), key=lambda x: -x[1])[:5]:
        last_pred = Prediction.query.filter_by(username=uname).order_by(Prediction.timestamp.desc()).first()
        top_users_list.append({
            "username": uname,
            "predictions": count,
            "last_active": last_pred.timestamp.strftime("%Y-%m-%d %H:%M") if last_pred else "-"
        })

    # -------- Last 7 days prediction trend --------
    today = datetime.utcnow().date()
    last_7_days_labels = []
    last_7_days_data = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        last_7_days_labels.append(d.strftime("%b %d"))
        count = sum(1 for p in all_preds if p.timestamp.date() == d)
        last_7_days_data.append(count)

    # -------- 24h counts --------
    yesterday = datetime.utcnow() - timedelta(hours=24)

    return render_template("admin.html",
        user=session["user"],
        users=users,
        logins=logins,
        predictions=predictions,
        total_users=User.query.count(),
        total_logins=LoginLog.query.filter_by(action="login").count(),
        total_signups=LoginLog.query.filter_by(action="signup").count(),
        total_predictions=Prediction.query.count(),
        total_feedback=len(all_fb),
        feedback_means=feedback_means,
        overall_mean=overall_mean,
        interpretation=interpretation,
        high_risk_count=high_risk_count,
        low_risk_count=low_risk_count,
        top_users=top_users_list,
        last_7_days_labels=last_7_days_labels,
        last_7_days_data=last_7_days_data,
        logins_24h=LoginLog.query.filter(LoginLog.timestamp >= yesterday, LoginLog.action == "login").count(),
        signups_24h=LoginLog.query.filter(LoginLog.timestamp >= yesterday, LoginLog.action == "signup").count(),
        predictions_24h=Prediction.query.filter(Prediction.timestamp >= yesterday).count())


@app.route("/admin/delete_user/<username>", methods=["POST"])
@admin_required
def admin_delete_user(username):
    if username == "admin":
        flash("Cannot delete the primary admin.", "error")
        return redirect(url_for("admin_dashboard"))
    user = User.query.filter_by(username=username).first()
    if user:
        Prediction.query.filter_by(username=username).delete()
        Feedback.query.filter_by(username=username).delete()
        LoginLog.query.filter_by(username=username).delete()
        db.session.delete(user)
        db.session.commit()
        flash(f"User '{username}' deleted.", "success")
    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    app.run(host=os.environ.get("FLASK_HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", 5000)), debug=False)