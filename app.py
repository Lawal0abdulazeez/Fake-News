from flask import Flask, render_template, request
import joblib
import pandas as pd
from werkzeug.urls import quote  # Import quote instead of url_quote

app = Flask(__name__)

# Load the pickled XGBoost model
with open('xgb_model.pkl', 'rb') as model_file:
    loaded_xgb_model = joblib.load(model_file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['text_input']

        # Make predictions on new data
        predictions = loaded_xgb_model.predict([text_input])

        # Display the prediction result
        result = 'Fake' if predictions[0] == 1 else 'True'

        return render_template('result.html', result=result, text_input=text_input)

if __name__ == '__main__':
    app.run(debug=True)
