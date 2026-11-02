from pathlib import Path
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from config import BASE_DIR, MODEL_INFO_PATH
from database.db import db, User, Recommendation
from ml.predict import load_model, predict_top_crops, FEATURES

app = Flask(__name__)
app.config.from_object("config")
db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Please log in to view your dashboard."
login_manager.init_app(app)

with app.app_context():
    (BASE_DIR / "database").mkdir(exist_ok=True)
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def get_model_info():
    if MODEL_INFO_PATH.exists():
        return json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    return None

@app.route("/")
def index():
    info = get_model_info()
    return render_template("index.html", info=info)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 8:
            flash("Enter your name, a valid email and a password of at least 8 characters.", "danger")
            return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "warning")
            return render_template("register.html")
        user = User(name=name, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Account created successfully.", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Incorrect email or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    history = Recommendation.query.filter_by(user_id=current_user.id).order_by(Recommendation.created_at.desc()).limit(10).all()
    info = get_model_info()
    return render_template("dashboard.html", history=history, info=info)

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "GET":
        return render_template("recommend.html")
    try:
        values = {}
        for key in FEATURES:
            values[key] = float(request.form.get(key, ""))
            if not (values[key] == values[key]) or abs(values[key]) == float("inf"):
                raise ValueError(f"{key} must be a finite number.")
        limits = {
            "N": (0, 200), "P": (0, 200), "K": (0, 250),
            "temperature": (-5, 60), "humidity": (0, 100),
            "ph": (0, 14), "rainfall": (0, 1000)
        }
        for key, (low, high) in limits.items():
            if not low <= values[key] <= high:
                raise ValueError(f"{key} must be between {low} and {high}.")
        model = load_model()
        ranked = predict_top_crops(model, values)
        record = Recommendation(
            user_id=current_user.id if current_user.is_authenticated else None,
            nitrogen=values["N"], phosphorus=values["P"], potassium=values["K"],
            temperature=values["temperature"], humidity=values["humidity"],
            ph=values["ph"], rainfall=values["rainfall"],
            predicted_crop=ranked[0]["crop"]
        )
        db.session.add(record)
        db.session.commit()
        return render_template("result.html", values=values, ranked=ranked, record=record)
    except (ValueError, TypeError) as exc:
        flash(str(exc), "danger")
    except FileNotFoundError as exc:
        flash(str(exc), "danger")
    return render_template("recommend.html"), 400

@app.route("/about")
def about():
    return render_template("about.html")

@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True)
