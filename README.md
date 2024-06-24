# Weather Prediction API

Simple prediction project for studies, hopefully i will pass :D

This project provides a Flask-based API for predicting rain based on historical weather data using a Random Forest classifier. It includes scripts for training the model and exposing predictions via an API endpoint.

## Installation

### Prerequisites

- Python
- pip (Python package installer)

### Dependencies

Install the required Python packages using `pip`. It's recommended to do this in a virtual environment to manage dependencies cleanly.

```bash
# Clone the repository
git clone <repository_url>
cd weather-prediction-api

# Create and activate a virtual environment (optional but recommended)
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies Installed

- Flask: Web framework for building the API.
- scikit-learn: Machine learning library for training the Random Forest model.
- pandas: Data manipulation and analysis library.
- numpy: Mathematical functions library used by pandas and scikit-learn.
- pickle: Serialization library for saving and loading the trained model.

## Running the App

### Training the Model

Before running the API, you need to train the machine learning model. Follow these steps:

1. **Prepare the Data**:

   - Place your weather data CSV file (`weatherAUS.csv`) under the `data` directory.

2. **Train the Model**:
   - Run the training script to preprocess the data, train the model, and save it to disk.

```bash
python ./src/train_model.py
```

3. **Save the Trained Model**:
   - After successful training, a file named `weather_model.pkl` will be created in the project directory.

### Starting the API Server

Once the model is trained and saved, you can start the Flask API server to expose predictions.

```bash
python ./src/server.py
```

The API server will start locally on `http://127.0.0.1:5000/`.

## API Documentation

### Endpoint

- **Endpoint**: `/predict`
- **Method**: `GET`
- **Parameters**: None (uses the latest day's data from `weatherAUS.csv`)
- **Response**: JSON object with the predicted value of 'RainTomorrow'

### Example Usage

Make a `GET` request to `/predict` to get the prediction for the next day's rain.

#### cURL Example

```bash
curl http://127.0.0.1:5000/predict
```

#### Python Example (using requests library)

```python
import requests

url = 'http://127.0.0.1:5000/predict'
response = requests.get(url)
data = response.json()

print(f"Prediction: {data['prediction']}")
```

### Notes

- The API uses the last recorded day's data from `weatherAUS.csv` to predict whether it will rain the next day.
- Categorical variables in the dataset are encoded using `LabelEncoder`.
- The model (`weather_model.pkl`) is deserialized from disk when the API server starts.
