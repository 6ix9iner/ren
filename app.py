from flask import Flask, request, render_template_string
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Initialize the Flask app
app = Flask(__name__)

# Define the features
numerical_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Load the trained model
model = joblib.load('best_xgb_insurance_model.pkl')

# Define the HTML template with CSS embedded
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Cost Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 400px;
            padding: 20px;
            background-color: white;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        label {
            font-weight: bold;
            margin-top: 10px;
        }
        input[type="number"], select {
            width: 100%;
            padding: 8px;
            margin-top: 5px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #4CAF50;
            border: none;
            color: white;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Insurance Cost Predictor</h1>
        <form action="/predict" method="post">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" required>

            <label for="sex">Sex</label>
            <select id="sex" name="sex" required>
                <option value="male">Male</option>
                <option value="female">Female</option>
            </select>

            <label for="bmi">BMI</label>
            <input type="number" id="bmi" name="bmi" step="0.1" required>

            <label for="children">Children</label>
            <input type="number" id="children" name="children" required>

            <label for="smoker">Smoker</label>
            <select id="smoker" name="smoker" required>
                <option value="yes">Yes</option>
                <option value="no">No</option>
            </select>

            <label for="region">Region</label>
            <select id="region" name="region" required>
                <option value="northeast">Northeast</option>
                <option value="northwest">Northwest</option>
                <option value="southeast">Southeast</option>
                <option value="southwest">Southwest</option>
            </select>

            <button type="submit">Predict</button>
        </form>
    </div>
</body>
</html>
'''

# Define the Flask route for the main page
@app.route('/')
def home():
    return render_template_string(html_template)

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Create a DataFrame from input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        # Preprocess the input data
        processed_input = preprocessor.transform(input_data)

        # Perform prediction
        prediction = model.predict(processed_input)[0]

        # Return the prediction as a response
        return f"<h2>Predicted Insurance Charge: ${prediction:.2f}</h2>"

    except Exception as e:
        return f"<h3>There was an error: {str(e)}</h3>"

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
