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
    return render_template('index.html')

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

    # Render the result page with prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
