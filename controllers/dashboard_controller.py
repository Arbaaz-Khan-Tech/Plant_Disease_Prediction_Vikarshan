from flask import jsonify, request
from datetime import datetime, timedelta
from bson import ObjectId

class DashboardController:
    def __init__(self, predictions_collection):
        self.predictions = predictions_collection

    def get_user_predictions(self, user_email):
        try:
            # Get predictions for the last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            predictions = list(self.predictions.find({
                'user_email': user_email,
                'created_at': {'$gte': thirty_days_ago}
            }).sort('created_at', -1))
            
            # Convert ObjectId to string for JSON serialization
            for pred in predictions:
                pred['_id'] = str(pred['_id'])
                pred['created_at'] = pred['created_at'].isoformat()
            
            return jsonify(predictions), 200
            
        except Exception as e:
            return jsonify({'error': 'Error fetching predictions'}), 500

    def save_prediction(self, user_email, prediction_data):
        try:
            prediction = {
                'user_email': user_email,
                'prediction': prediction_data['prediction'],
                'confidence': prediction_data['confidence'],
                'image_path': prediction_data.get('image_path'),
                'created_at': datetime.utcnow()
            }
            
            result = self.predictions.insert_one(prediction)
            prediction['_id'] = str(result.inserted_id)
            prediction['created_at'] = prediction['created_at'].isoformat()
            
            return jsonify(prediction), 201
            
        except Exception as e:
            return jsonify({'error': 'Error saving prediction'}), 500

    def get_prediction_stats(self, user_email):
        try:
            # Get total predictions
            total_predictions = self.predictions.count_documents({'user_email': user_email})
            
            # Get predictions for the last 7 days
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            recent_predictions = self.predictions.count_documents({
                'user_email': user_email,
                'created_at': {'$gte': seven_days_ago}
            })
            
            # Get average confidence
            pipeline = [
                {'$match': {'user_email': user_email}},
                {'$group': {'_id': None, 'avg_confidence': {'$avg': '$confidence'}}}
            ]
            avg_confidence = list(self.predictions.aggregate(pipeline))
            avg_confidence = avg_confidence[0]['avg_confidence'] if avg_confidence else 0
            
            return jsonify({
                'total_predictions': total_predictions,
                'recent_predictions': recent_predictions,
                'average_confidence': round(avg_confidence, 2)
            }), 200
            
        except Exception as e:
            return jsonify({'error': 'Error fetching statistics'}), 500 