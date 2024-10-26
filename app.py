from flask import Flask, render_template, request
import numpy as np
import joblib

# Load the model
model = joblib.load('best_xgb_insurance_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the home route to display the input form
@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Insurance Cost Predictor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: url('https://www.google.com/imgres?q=images%20for%20web%20background%20download&imgurl=https%3A%2F%2Fplus.unsplash.com%2Fpremium_photo-1673480195911-3075a87738b0%3Ffm%3Djpg%26q%3D60%26w%3D3000%26ixlib%3Drb-4.0.3%26ixid%3DM3wxMjA3fDB8MHxzZWFyY2h8MXx8d2ViJTIwYmFja2dyb3VuZHxlbnwwfHwwfHx8MA%253D%253D&imgrefurl=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fweb-background&docid=ZflMew8d06L0PM&tbnid=V7-TiwsRk-hdzM&vet=12ahUKEwiGsb2o3KyJAxU9WkEAHUm0DNkQM3oECGcQAA..i&w=3000&h=2000&hcb=2&ved=2ahUKEwiGsb2o3KyJAxU9WkEAHUm0DNkQM3oECGcQAA') no-repeat center center fixed;
                background-size: cover;
                color: #333;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background-color: rgba(255, 255, 255, 0.8);
                padding: 20px;
                border-radius: 8px;
                width: 300px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
            }
            input, select {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            button {
                width: 100%;
                padding: 10px;
                background-color: #007BFF;
                border: none;
                color: white;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Insurance Cost Predictor</h2>
            <form action="/predict" method="post">
                <div class="form-group">
                    <label for="age">Age:</label>
                    <input type="number" id="age" name="age" required>
                </div>
                <div class="form-group">
                    <label for="sex">Sex:</label>
                    <select id="sex" name="sex" required>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="bmi">BMI:</label>
                    <input type="number" id="bmi" name="bmi" step="0.1" required>
                </div>
                <div class="form-group">
                    <label for="children">Number of Children:</label>
                    <input type="number" id="children" name="children" required>
                </div>
                <div class="form-group">
                    <label for="smoker">Smoker:</label>
                    <select id="smoker" name="smoker" required>
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="region">Region:</label>
                    <select id="region" name="region" required>
                        <option value="northeast">Northeast</option>
                        <option value="northwest">Northwest</option>
                        <option value="southeast">Southeast</option>
                        <option value="southwest">Southwest</option>
                    </select>
                </div>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    """

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = 1 if request.form['smoker'] == 'yes' else 0
    sex_female = 1 if request.form['sex'] == 'female' else 0

    # Encode the region as one-hot
    region = request.form['region']
    region_northeast = region_northwest = region_southeast = region_southwest = 0
    if region == 'northeast':
        region_northeast = 1
    elif region == 'northwest':
        region_northwest = 1
    elif region == 'southeast':
        region_southeast = 1
    elif region == 'southwest':
        region_southwest = 1

    # Prepare input for the model
    input_features = np.array([[age, sex_female, bmi, children, smoker, region_northeast, region_northwest, region_southeast, region_southwest]])

    # Make the prediction
    prediction = model.predict(input_features)[0]

    # Render the result page with prediction
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
                padding: 20px;
                border-radius: 8px;
                background-color: rgba(255, 255, 255, 0.8);
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
        </style>
    </head>
    <body>
        <div class="result-container">
            <h2>Prediction Result</h2>
            <p>The predicted insurance cost is: <strong>${{prediction:.2f}}</strong></p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, port=5001)

            
