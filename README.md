# Plant Disease Prediction Web Application

## Overview
This project is a web-based plant disease prediction application that helps farmers and gardeners identify diseases in their plants using machine learning. The core machine learning model and training code are based on the work by [AHMEDSANA](https://github.com/AHMEDSANA/Plant-Disease-Detection).

## Credits
- Original Model & Training Code: [AHMEDSANA's Plant Disease Detection](https://github.com/AHMEDSANA/Plant-Disease-Detection)
- The machine learning model and training methodology are not owned by me and full credit goes to the original author.

## My Contributions
I have enhanced the original project by adding:
1. A modern web interface using Flask as the backend server
2. User authentication system with login/signup functionality
3. A comprehensive dashboard for farm management, including:
   - Weather monitoring
   - Soil health tracking
   - Crop health status
   - Yield forecasting
   - Resource usage analytics
   - Task management
   - Equipment status monitoring
   - Market price tracking

## Features
- **User Authentication**
  - Secure login and signup system
  - JWT-based authentication
  - Password hashing for security

- **Disease Prediction**
  - Upload plant images for disease detection
  - Get instant predictions using the pre-trained model
  - View prediction history

- **Farm Management Dashboard**
  - Real-time weather updates
  - Soil health monitoring
  - Crop status tracking
  - Interactive charts and analytics
  - Task management system
  - Equipment status monitoring
  - Market price tracking

## Technologies Used
- Backend: Flask (Python)
- Frontend: HTML, Tailwind CSS, JavaScript
- Authentication: JWT
- Database: MongoDB
- Charts: Chart.js
- UI Components: Font Awesome icons

## Setup and Installation
1. Clone the repository
2. Install required packages:
   ```
   pip install flask pymongo python-dotenv pyjwt
   ```
3. Set up environment variables in `.env` file:
   ```
   MONGODB_URI=your_mongodb_connection_string
   JWT_SECRET=your_jwt_secret
   ```
4. Run the application:
   ```
   python app.py
   ```

## Usage
1. Register/Login to access the dashboard
2. Upload plant images for disease detection
3. View predictions and historical data
4. Use the dashboard to monitor farm metrics
5. Manage tasks and equipment status
6. Track market prices

## Disclaimer
The machine learning model and training code are the intellectual property of [AHMEDSANA](https://github.com/AHMEDSANA/Plant-Disease-Detection). This project builds upon their work by adding a web interface and additional features for farm management.

## License
Please refer to the original project's license at [AHMEDSANA's Repository](https://github.com/AHMEDSANA/Plant-Disease-Detection) for the model and training code. The additional features and UI enhancements are available for use under standard open-source terms. 