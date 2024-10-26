from flask import Flask, request
import numpy as np
import joblib

# Load the model
model = joblib.load('best_xgb_insurance_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the home route with embedded HTML and CSS
@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Insurance Cost Predictor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-image: url('data:image/jpeg;base64,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');
                background-size: cover;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background: rgba(255, 255, 255, 0.8);
                padding: 20px;
                border-radius: 8px;
                width: 300px;
                text-align: center;
            }
            input, button {
                margin-top: 10px;
                padding: 10px;
                width: 100%;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            button {
                background-color: #4CAF50;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Insurance Cost Predictor</h2>
            <form action="/predict" method="post">
                <input type="text" name="age" placeholder="Age" required><br>
                <input type="text" name="bmi" placeholder="BMI" required><br>
                <input type="text" name="children" placeholder="Number of Children" required><br>
                <select name="smoker" required>
                    <option value="">Smoker?</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                </select><br>
                <select name="sex" required>
                    <option value="">Gender</option>
                    <option value="female">Female</option>
                    <option value="male">Male</option>
                </select><br>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    """
    return html

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = 1 if request.form['smoker'] == 'yes' else 0
    sex_female = 1 if request.form['sex'] == 'female' else 0

    # Prepare input for the model
    input_features = np.array([[age, bmi, children, smoker, sex_female]])

    # Make the prediction
    prediction = model.predict(input_features)[0]

    # Return the result with embedded HTML
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            .result-container {{
                text-align: center;
                background: rgba(255, 255, 255, 0.9);
                padding: 20px;
                border-radius: 8px;
            }}
            a {{
                display: inline-block;
                margin-top: 20px;
                color: #4CAF50;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="result-container">
            <h2>Predicted Insurance Cost</h2>
            <p>Your estimated insurance cost is: <strong>${prediction:.2f}</strong></p>
            <a href="/">Go Back</a>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)
