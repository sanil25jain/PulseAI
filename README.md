# PulseAI 🫀

**PulseAI** is an intelligent web application designed to predict the likelihood of heart disease based on clinical parameters using Machine Learning. It also features an integrated AI Chatbot to assist users with medical queries and heart health advice.

[**View Live Demo**](https://pulseai-app.onrender.com) ---

## ✨ Key Functionalities

### 1. Heart Disease Prediction
* **Input:** Accepts 13 standard clinical features (Age, Sex, Chest Pain Type, BP, Cholesterol, ECG, etc.).
* **Algorithm:** Utilizes a trained **Logistic Regression** model (`lr.pkl`) scaled with **StandardScaler** (`sc.pkl`) to analyze user data.
* **Output:** Provides an immediate diagnosis (Presence or Absence of Heart Disease) with visual indicators.

### 2. AI Medical Assistant (Chatbot)
* **Powered by:** Groq API & Llama 3 (Llama-3.3-70b-versatile).
* **Interface:** Includes a floating chatbot widget for easy access.
* **Context-Aware:** Specially engineered to answer questions regarding heart health, diet, and medical terminology.
* **Guardrails:** Politely refuses to answer non-medical questions (e.g., sports, coding) to maintain a strict focus on health.

### 3. User System
* **Authentication:** Secure Login and Registration system.
* **Security:** Password hashing using **Werkzeug** and session management via **Flask-Login**.
* **Database:** Securely stores user credentials and history.

### 4. Responsive UI
* **Design:** Modern, clean interface built with **Tailwind CSS**.
* **Structure:** Vertical, card-based layout optimized for easy data entry on both mobile and desktop devices.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Backend Framework** | Flask (Flask-SQLAlchemy, Flask-Login) |
| **Data Science & ML** | Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn |
| **Generative AI** | Groq API (Llama-3.3-70b-versatile) |
| **Frontend** | HTML5 (Jinja2), Tailwind CSS, JavaScript |
| **Database** | PostgreSQL (Production), SQLite (Local) |
| **Server & Deployment** | Gunicorn (WSGI Server), Render Cloud |
| **Security** | Werkzeug (Password Hashing) |

---