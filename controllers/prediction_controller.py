from flask import jsonify, request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

class PredictionController:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.upload_folder = 'uploads'
        self.allowed_extensions = {'png', 'jpg', 'jpeg'}
        
        # Create uploads directory if it doesn't exist
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def preprocess_image(self, image_path):
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to match model input size
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self):
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and self.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(self.upload_folder, filename)
            file.save(filepath)
            
            try:
                # Preprocess the image
                processed_image = self.preprocess_image(filepath)
                
                # Make prediction
                prediction = self.model.predict(processed_image)
                predicted_class = np.argmax(prediction[0])
                confidence = float(np.max(prediction[0]))
                
                # Clean up the uploaded file
                os.remove(filepath)
                
                return jsonify({
                    'prediction': int(predicted_class),
                    'confidence': confidence,
                    'message': 'Prediction successful'
                }), 200
                
            except Exception as e:
                return jsonify({'error': 'Error processing image'}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400 