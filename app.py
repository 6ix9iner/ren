from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_xgb_insurance_model.pkl')

# Define feature preprocessing
numerical_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# Define the transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))  # drop='first' to avoid dummy variable trap
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor on some dummy data
dummy_data = pd.DataFrame({
    'age': [30],
    'sex': ['female'],
    'bmi': [25.0],
    'children': [1],
    'smoker': ['no'],
    'region': ['southeast']
})
preprocessor.fit(dummy_data)  # Fit preprocessor once to avoid errors later

# Define a function for predictions
def predict_insurance(age, sex, bmi, children, smoker, region):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Transform the input data
    transformed_data = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(transformed_data)
    
    return prediction[0]

# HTML and CSS embedded into the Flask app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1 {
            color: #333;
        }
        form {
            background: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        label {
            display: block;
            margin-bottom: 8px;
        }
        input, select {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            background-color: #28a745;
            color: white;
            padding: 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #218838;
        }
    </style>
</head>
<body>
    <h1>Medical Insurance Cost Predictor</h1>
    <form action="/predict" method="POST">
        <label for="age">Age:</label>
        <input type="number" id="age" name="age" required>
        
        <label for="sex">Sex:</label>
        <select id="sex" name="sex" required>
            <option value="male">Male</option>
            <option value="female">Female</option>
        </select>
        
        <label for="bmi">BMI:</label>
        <input type="number" step="0.1" id="bmi" name="bmi" required>
        
        <label for="children">Number of Children:</label>
        <input type="number" id="children" name="children" required>
        
        <label for="smoker">Smoker:</label>
        <select id="smoker" name="smoker" required>
            <option value="yes">Yes</option>
            <option value="no">No</option>
        </select>
        
        <label for="region">Region:</label>
        <select id="region" name="region" required>
            <option value="northeast">Northeast</option>
            <option value="northwest">Northwest</option>
            <option value="southeast">Southeast</option>
            <option value="southwest">Southwest</option>
        </select>
        
        <button type="submit">Predict</button>
    </form>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        # Get prediction
        predicted_charge = predict_insurance(age, sex, bmi, children, smoker, region)
        return f'Predicted Insurance Charge: ${predicted_charge:.2f}'
    except Exception as e:
        return f'There was an error: {e}'

if __name__ == '__main__':
    app.run(debug=True)
