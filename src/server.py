from flask import Flask, request, jsonify
import pickle
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('weather_model.pkl', 'rb') as f:
    model = pickle.load(f)  # Deserialize the trained model from file

# Load and preprocess the data
data = pd.read_csv('./data/weatherAUS.csv')  # Read the weather data into a DataFrame

# Fill NA values and encode categorical data
data = data.fillna(method='ffill')  # Fill missing values using forward fill
le = LabelEncoder()  # Initialize LabelEncoder for encoding categorical variables
for column in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']:
    data[column] = le.fit_transform(data[column])  # Encode categorical columns

app = Flask(__name__)  # Initialize Flask application

@app.route('/predict', methods=['GET'])  # Define endpoint '/predict' for GET requests
def predict():
    # Prepare the input data for the next day prediction
    last_day = data.iloc[-1]  # Get the last day's data
    next_day = last_day.copy()  # Create a copy of the last day's data
    next_day['Date'] = (datetime.datetime.strptime(last_day['Date'], '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')  # Advance the date by one day

    # Predict for the next day
    input_data = next_day.drop(labels=['Date', 'RainTomorrow']).values.reshape(1, -1)  # Prepare input data for prediction
    prediction = model.predict(input_data)  # Use the trained model to predict

    # Decode the prediction
    prediction = le.inverse_transform(prediction)[0]  # Inverse transform to get categorical label

    return jsonify({'willItRainTomorrow': prediction})  # Return prediction as JSON response

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask application in debug mode
