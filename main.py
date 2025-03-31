
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import requests
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/finance_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    income = db.Column(db.Float, nullable=False)
    expenses = db.relationship('Expense', backref='user', lazy=True)
# Expense Model
class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
# Initialize Database
with app.app_context():
    db.create_all()
data = pd.DataFrame({
    'amount': np.random.randint(10, 500, 100),
    'category': np.random.choice(['Food', 'Rent', 'Travel', 'Shopping'], 100),
    'income': np.random.randint(2000, 10000, 100)
})
# Categorization Model
X = data[['amount', 'income']]
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
joblib.dump(clf, 'spending_classifier.pkl')
# Expense Prediction Model
model = LinearRegression()
model.fit(X_train, X_train['amount'])
joblib.dump(model, 'expense_predictor.pkl')
#Financial API Integration
ALPHA_VANTAGE_API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'

@app.route('/stock/<symbol>', methods=['GET'])
def get_stock(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    return jsonify(response.json())
@app.route('/')
def home():
    return "AI-Powered Personal Finance Advisor Backend Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model = joblib.load('expense_predictor.pkl')
    prediction = model.predict(np.array([[data['amount'], data['income']]]))
    return jsonify({'predicted_expense': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
