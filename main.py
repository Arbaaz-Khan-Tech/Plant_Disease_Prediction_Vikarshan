from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import os
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from dotenv import load_dotenv
import jwt
from functools import wraps
import datetime

# Load environment variables
load_dotenv()

# MongoDB setup
client = MongoClient(os.getenv('MONGODB_URI'))
db = client.vikarshan
users = db.users

# Define your CNN model (same as you defined above)
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super(CNNClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Flask app setup
app = Flask(__name__)

# Load your class names and model
class_names = [
    'Pepper_bell__Bacterial_spot',
    'Pepper_bell__healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

model = CNNClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load('final_model.pth', map_location=torch.device('cpu')))
model.eval()

# Image pre-processing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_disease(image):
    image_tensor = transform(image).unsqueeze(0)
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_prob, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence = predicted_prob.item() * 100
    return predicted_class, confidence

# JWT token required decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            return redirect(url_for('auth'))
        try:
            data = jwt.decode(token, os.getenv('JWT_SECRET'), algorithms=["HS256"])
            current_user = users.find_one({'email': data['email']})
            if not current_user:
                return redirect(url_for('auth'))
        except:
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated

# Routes for the web app
@app.route('/')
@token_required
def index():
    return render_template('index.html')

@app.route('/auth')
def auth():
    return render_template('auth.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Check if user already exists
    if users.find_one({'email': data['email']}):
        return jsonify({'error': 'Email already registered'}), 400
    
    # Validate password match
    if data['password'] != data['confirmPassword']:
        return jsonify({'error': 'Passwords do not match'}), 400
    
    # Create new user
    new_user = {
        'firstName': data['firstName'],
        'lastName': data['lastName'],
        'email': data['email'],
        'password': generate_password_hash(data['password']),
        'created_at': datetime.datetime.utcnow()
    }
    
    try:
        users.insert_one(new_user)
        
        # Generate JWT token
        token = jwt.encode({
            'email': new_user['email'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
        }, os.getenv('JWT_SECRET'))
        
        response = jsonify({'message': 'Registration successful'})
        response.set_cookie('token', token, httponly=True, secure=True, samesite='Strict')
        return response, 200
    
    except Exception as e:
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = users.find_one({'email': data['email']})
    
    if user and check_password_hash(user['password'], data['password']):
        # Generate JWT token
        token = jwt.encode({
            'email': user['email'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
        }, os.getenv('JWT_SECRET'))
        
        response = jsonify({'message': 'Login successful'})
        response.set_cookie('token', token, httponly=True, secure=True, samesite='Strict')
        return response, 200
    
    return jsonify({'error': 'Invalid email or password'}), 401

@app.route('/logout')
def logout():
    response = redirect(url_for('auth'))
    response.delete_cookie('token')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream).convert('RGB')
            predicted_class, confidence = predict_disease(image)
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence
            })
    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
