from flask import jsonify, request, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import os
from functools import wraps

class AuthController:
    def __init__(self, users_collection):
        self.users = users_collection

    def token_required(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.cookies.get('token')
            if not token:
                return redirect(url_for('auth'))
            try:
                data = jwt.decode(token, os.getenv('JWT_SECRET'), algorithms=["HS256"])
                current_user = self.users.find_one({'email': data['email']})
                if not current_user:
                    return redirect(url_for('auth'))
            except:
                return redirect(url_for('auth'))
            return f(*args, **kwargs)
        return decorated

    def register(self):
        data = request.get_json()
        
        # Check if user already exists
        if self.users.find_one({'email': data['email']}):
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
            self.users.insert_one(new_user)
            
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

    def login(self):
        data = request.get_json()
        user = self.users.find_one({'email': data['email']})
        
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

    def logout(self):
        response = redirect(url_for('auth'))
        response.delete_cookie('token')
        return response 