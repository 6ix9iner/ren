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
                background-image: url('https://example.com/your-background.jpg');
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
