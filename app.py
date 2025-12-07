from flask import (
    Flask, request, render_template, redirect, url_for, flash, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pickle
import os
import time
from groq import Groq # Import the Groq library

# --- Load Models ---
try:
    with open('lr.pkl', 'rb') as file:
        lr = pickle.load(file)
    with open('sc.pkl', 'rb') as scaler_file:
        sc = pickle.load(scaler_file)
except Exception as e:
    print(f"\n[ERROR] Could not load models: {e}\n")
    lr = None
    sc = None

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')

# --- Database Config ---
database_url = os.environ.get('DATABASE_URL')
if database_url:
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Groq AI Configuration ---
# You need a GROQ_API_KEY instead of GEMINI_API_KEY now.
# Get one for free at https://console.groq.com/keys
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# --- Debug: Print Key Status on Startup ---
if GROQ_API_KEY:
    print(f"DEBUG: Groq API Key found. Starts with: {GROQ_API_KEY[:4]}... (Length: {len(GROQ_API_KEY)})")
else:
    print("DEBUG: No GROQ_API_KEY found in environment variables.")

groq_client = None

def get_groq_client():
    """
    Initializes the Groq client.
    """
    global groq_client
    if groq_client:
        return groq_client
        
    if not GROQ_API_KEY:
        return None

    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        return groq_client
    except Exception as e:
        print(f"Error configuring Groq: {e}")
        return None

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

@app.cli.command('init-db')
def init_db_command():
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
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username exists.', 'warning')
            return render_template('register.html')
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/predictor')
@login_required
def predictor():
    return render_template('predictor.html', prediction_text="")

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not lr or not sc:
        flash('Models not loaded.', 'danger')
        return render_template('predictor.html', prediction_text="")
    try:
        input_data = [float(x) for x in request.form.values()]
        # We only take the first 13 inputs in case extra data was sent
        if len(input_data) < 13: 
             pass 

        clinical_features = input_data[:13]
        
        if len(clinical_features) != 13:
            flash(f'Error: Expected 13 features, got {len(clinical_features)}.', 'danger')
            return render_template('predictor.html', prediction_text="")

        input_data_as_numpy_array = np.asarray(clinical_features)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        std_data = sc.transform(input_data_reshaped)
        prediction = lr.predict(std_data)
        
        if prediction[0] == 1:
            result = 'Patient Diagnosed With Heart Disease'
            cls = 'text-red-600'
        else:
            result = 'CONGRATULATIONS! Patient Not Diagnosed With Heart Disease'
            cls = 'text-green-600'
        return render_template('predictor.html', prediction_text=result, prediction_class=cls)
    except Exception as e:
        flash(f'Error: {e}', 'danger')
        return render_template('predictor.html', prediction_text="")

# --- Chatbot API Route (GROQ VERSION) ---
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    client = get_groq_client()
    
    if not client:
        # Check if they forgot to set the new key
        return jsonify({'response': "System offline: GROQ_API_KEY is missing."})

    user_message = request.json.get('message')
    
    # System Instruction for context
    system_prompt = """
    You are PulseAI Assistant, a medical AI focused ONLY on heart health.
    1. Answer questions about heart disease, diet, exercise, and medical terms helpfully.
    2. Politely REFUSE to answer questions about unrelated topics (sports, coding, politics, etc.).
    3. Keep answers concise (max 3-4 sentences).
    """
    
    try:
        # Create completion with Llama 3.3 (Updated Model)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.3-70b-versatile", # UPDATED: The previous model was decommissioned
            temperature=0.5,
            max_tokens=300,
        )
        
        # Extract response
        ai_response = chat_completion.choices[0].message.content
        return jsonify({'response': ai_response})
        
    except Exception as e:
        print(f"Groq Error: {e}")
        error_str = str(e)
        if "401" in error_str:
             return jsonify({'response': "Error: Invalid API Key. Please check your GROQ_API_KEY setting."})
        return jsonify({'response': f"I'm having trouble connecting. Error: {str(e)}"})

# --- Main execution ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)