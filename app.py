from flask import (
    Flask, request, render_template, redirect, url_for, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pickle
import os

# --- Load Models with Error Handling ---
# We use try/except to give a clear error if the files are missing.
try:
    with open('lr.pkl', 'rb') as file:
        lr = pickle.load(file)
except FileNotFoundError:
    print("\n[ERROR] 'lr.pkl' file not found.")
    print("Make sure 'lr.pkl' is in the same directory as 'app.py'\n")
    lr = None
except Exception as e:
    print(f"\n[ERROR] Could not load 'lr.pkl': {e}\n")
    lr = None

try:
    with open('sc.pkl', 'rb') as scaler_file:
        sc = pickle.load(scaler_file)
except FileNotFoundError:
    print("\n[ERROR] 'sc.pkl' file not found.")
    print("Make sure 'sc.pkl' is in the same directory as 'app.py'\n")
    sc = None
except Exception as e:
    print(f"\n[ERROR] Could not load 'sc.pkl': {e}\n")
    sc = None

# --- App Initialization ---
app = Flask(__name__)

# --- Configuration (Local SQLite) ---
# Secret key for session management
app.config['SECRET_KEY'] = 'a_very_secret_key_for_local_testing'

# Database configuration for local SQLite
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Database & Login Manager Setup ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to login page if user is not logged in
login_manager.login_message_category = 'warning' # Flash message category

# --- User Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Database Initializer Command ---
# This function will create the database tables
@app.cli.command('init-db')
def init_db_command():
    """Creates the database tables."""
    with app.app_context():
        db.create_all()
    print('Initialized the database.')

# --- Routes ---

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('predictor'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('predictor'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('predictor'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'warning')
            return render_template('register.html')

        # Create new user
        new_user = User(username=username)
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred: {e}', 'danger')

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/predictor')
@login_required
def predictor():
    """Renders the predictor page."""
    return render_template('predictor.html', prediction_text="")

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handles the prediction logic from the form."""
    
    # Check if models loaded correctly
    if not lr or not sc:
        flash('Prediction models are not loaded. Please check server logs.', 'danger')
        return render_template('predictor.html', prediction_text="")

    try:
        # Get all values from the form
        input_data = [float(x) for x in request.form.values()]
        
        # Check if we have the correct number of features
        if len(input_data) != 13:
            flash(f'Error: Expected 13 features, but received {len(input_data)}.', 'danger')
            return render_template('predictor.html', prediction_text="")

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Standardize the data
        std_data = sc.transform(input_data_reshaped)
        
        # Make prediction
        prediction = lr.predict(std_data)
        
        # Format the result
        result_text = ''
        result_class = ''
        if prediction[0] == 1:
            result_text = 'Patient Diagnosed With Heart Disease'
            result_class = 'text-red-600'
        else:
            result_text = 'CONGRATULATIONS! Patient Not Diagnosed With Heart Disease'
            result_class = 'text-green-600'
            
        return render_template('predictor.html', prediction_text=result_text, prediction_class=result_class)
    
    except ValueError:
        flash('Invalid data submitted. Please check all fields.', 'danger')
        return render_template('predictor.html', prediction_text="")
    except Exception as e:
        # Catch other potential errors (e.g., shape mismatch in sc.transform)
        print(f"\n[ERROR] in /predict route: {e}\n")
        flash(f'An error occurred during prediction: {e}', 'danger')
        return render_template('predictor.html', prediction_text="")

# --- Main execution ---
if __name__ == "__main__":
    # --- Create tables ---
    # This block runs ONLY when you execute `python app.py`
    # It ensures the database tables exist, fixing the "no such table" error
    # for this specific run method.
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)
