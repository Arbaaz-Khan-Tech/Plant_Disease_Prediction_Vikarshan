from flask import Flask, render_template, request, jsonify, redirect, url_for
from pymongo import MongoClient
from dotenv import load_dotenv
import os

from controllers.auth_controller import AuthController
from controllers.prediction_controller import PredictionController
from controllers.dashboard_controller import DashboardController

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DB')]
users_collection = db.users
predictions_collection = db.predictions

# Initialize controllers
auth_controller = AuthController(users_collection)
prediction_controller = PredictionController('model/plant_disease_model.h5')
dashboard_controller = DashboardController(predictions_collection)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth')
def auth():
    return render_template('auth.html')

@app.route('/register', methods=['POST'])
def register():
    return auth_controller.register()

@app.route('/login', methods=['POST'])
def login():
    return auth_controller.login()

@app.route('/logout')
def logout():
    return auth_controller.logout()

@app.route('/predict', methods=['POST'])
@auth_controller.token_required
def predict():
    # Get user email from token
    token = request.cookies.get('token')
    data = jwt.decode(token, os.getenv('JWT_SECRET'), algorithms=["HS256"])
    user_email = data['email']
    
    # Get prediction from controller
    prediction_response = prediction_controller.predict()
    
    if prediction_response[1] == 200:
        # Save prediction to database
        prediction_data = prediction_response[0].get_json()
        dashboard_controller.save_prediction(user_email, prediction_data)
    
    return prediction_response

@app.route('/dashboard')
@auth_controller.token_required
def dashboard():
    # Get user email from token
    token = request.cookies.get('token')
    data = jwt.decode(token, os.getenv('JWT_SECRET'), algorithms=["HS256"])
    user_email = data['email']
    
    # Get user's predictions and stats
    predictions = dashboard_controller.get_user_predictions(user_email)
    stats = dashboard_controller.get_prediction_stats(user_email)
    
    return render_template('dashboard.html', 
                         predictions=predictions[0].get_json(),
                         stats=stats[0].get_json())

if __name__ == '__main__':
    app.run(debug=True) 