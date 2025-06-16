from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    sqft = float(request.form['sqft'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])

    # Create input DataFrame
    input_data = pd.DataFrame([[sqft, bedrooms, bathrooms]], 
                             columns=['sqft', 'bedrooms', 'bathrooms'])

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction = round(prediction, 2)

    return render_template('index.html', prediction=f'Predicted House Price: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
