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
import google.generativeai as genai

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

# --- Gemini AI Configuration (Lazy Loading) ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
_model = None 

def get_ai_model():
    """
    Initializes and returns the Gemini model.
    This runs ONLY when the first chat request comes in, preventing server timeout during startup.
    """
    global _model
    if _model:
        return _model
        
    if not GEMINI_API_KEY:
        print("Error: No GEMINI_API_KEY found.")
        return None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # 1. Ask Google what models are available
        print("--- Checking available AI models (Lazy Load) ---")
        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except Exception as e:
            print(f"Warning: Could not list models ({e}).")

        # 2. Select the best available model
        chosen_model_name = None
        priority_substrings = [
            'gemini-1.5-flash', 
            'gemini-1.5-pro',
            'gemini-1.0-pro',
            'gemini-pro'
        ]
        
        for priority in priority_substrings:
            for model_name in available_models:
                if priority in model_name and 'exp' not in model_name:
                    chosen_model_name = model_name
                    break
            if chosen_model_name:
                break
        
        if not chosen_model_name:
            # Fallback 1: First non-experimental
            for model_name in available_models:
                if 'exp' not in model_name and 'vision' not in model_name: 
                    chosen_model_name = model_name
                    break
        
        if not chosen_model_name:
            # Fallback 2: Hard default
            chosen_model_name = 'models/gemini-1.5-flash'

        print(f"SUCCESS: Initialized AI Model -> {chosen_model_name}")
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        _model = genai.GenerativeModel(model_name=chosen_model_name, 
                                      generation_config=generation_config)
        return _model

    except Exception as e:
        print(f"Error configuring Gemini: {e}")
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
        if len(input_data) != 13:
            flash(f'Error: Expected 13 features, got {len(input_data)}.', 'danger')
            return render_template('predictor.html', prediction_text="")

        input_data_as_numpy_array = np.asarray(input_data)
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

# --- Chatbot API Route ---
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    if not GEMINI_API_KEY:
        return jsonify({'response': "AI System is currently offline (API Key missing)."})
    
    # LAZY LOAD: We initialize the model here, the first time someone chats
    model = get_ai_model()
    
    if not model:
        return jsonify({'response': "AI Model initialization failed. Check server logs for details."})

    user_message = request.json.get('message')
    
    # Context prompt
    prompt_with_context = f"""
    You are PulseAI Assistant, a specialized medical AI focused ONLY on heart health.
    If the user asks about heart health, diet, or medical terms, answer helpfully.
    If the user asks about ANYTHING else, politely REFUSE.
    
    User Query: {user_message}
    """
    
    try:
        response = model.generate_content(prompt_with_context)
        return jsonify({'response': response.text})
    except Exception as e:
        print(f"Gemini Error: {e}")
        return jsonify({'response': f"I'm having trouble thinking right now. Error details: {str(e)}"})

# --- Main execution ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)