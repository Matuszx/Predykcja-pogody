import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load and preprocess the data
data = pd.read_csv('./data/weatherAUS.csv')  # Read the weather data into a DataFrame

# Fill NA values and encode categorical data
data = data.fillna(method='ffill')  # Fill missing values using forward fill
le = LabelEncoder()  # Initialize LabelEncoder for encoding categorical variables
for column in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']:
    data[column] = le.fit_transform(data[column])  # Encode categorical columns

# Define features and target
X = data.drop(columns=['Date', 'RainTomorrow'])  # Features: all columns except 'Date' and 'RainTomorrow'
y = data['RainTomorrow']  # Target variable: 'RainTomorrow'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize RandomForestClassifier
model.fit(X_train, y_train)  # Train the model using training data

# Save the model to disk
with open('weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)  # Serialize the trained model to a file
