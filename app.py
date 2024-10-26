from flask import Flask, render_template_string, request
import numpy as np
import joblib

# Load the model
model = joblib.load('best_xgb_insurance_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# HTML and CSS embedded within the app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Cost Predictor</title>
    <style>
        body {
            background-image: url('https://example.com/your-background-image.jpg'); /* Replace with your image URL */
            background-size: cover;
            font-family: Arial, sans-serif;
            color: white;
        }
        form {
            width: 300px;
            margin: auto;
            padding: 20px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
        }
        input, select {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: none;
        }
        button {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Predict Medical Insurance Cost</h1>
    <form action="/predict" method="POST">
        <input type="number" name="age" placeholder="Age" required>
        <input type="number" step="0.1" name="bmi" placeholder="BMI" required>
        <input type="number" name="children" placeholder="Number of Children" required>
        <select name="smoker" required>
            <option value="">Select Smoker Status</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
        </select>
        <select name="sex" required>
            <option value="">Select Sex</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
        </select>
        <select name="region" required>
            <option value="">Select Region</option>
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

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <style>
        body {
            background-image: url('https://example.com/your-background-image.jpg'); /* Replace with your image URL */
            background-size: cover;
            font-family: Arial, sans-serif;
            color: white;
        }
        .result {
            width: 300px;
            margin: auto;
            padding: 20px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            text-align: center;
        }
        button {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="result">
        <h1>Prediction Result</h1>
        <h2>Your predicted insurance cost is: ${{ prediction }}</h2>
        <a href="/">
            <button>Back</button>
        </a>
    </div>
</body>
</html>
"""

# Define the home route to display the input form
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smoker'] == 'yes' else 0
        sex_female = 1 if request.form['sex'] == 'female' else 0
        
        # Handle the region feature
        region_northwest = 1 if request.form['region'] == 'northwest' else 0
        region_northeast = 1 if request.form['region'] == 'northeast' else 0
        region_southeast = 1 if request.form['region'] == 'southeast' else 0
        region_southwest = 1 if request.form['region'] == 'southwest' else 0
        
        # Prepare input for the model
        input_features = np.array([[age, bmi, children, smoker, sex_female, 
                                    region_northwest, region_northeast, 
                                    region_southeast, region_southwest]])
        
        # Make the prediction
        prediction = model.predict(input_features)[0]

        # Render the result page with prediction
        return render_template_string(RESULT_TEMPLATE, prediction=prediction)
    except Exception as e:
        return f"<h3>There was an error: {str(e)}</h3>", 500

if __name__ == '__main__':
    app.run(debug=True)

       


       
