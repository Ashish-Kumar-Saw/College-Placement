from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the CGPA and IQ from the form
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    # Prepare data for prediction
    input_data = np.array([[cgpa, iq]])
    
    # Make prediction
    prediction = model.predict(input_data)

    # Prepare result message based on prediction
    if int(prediction[0]) == 1:
        result_message = "Placement ho jayega!"  # Assuming 1 means placement
    else:
        result_message = "Placement nahi hoga!"   # Assuming 0 means no placement

    return render_template('index.html', result=result_message)

if __name__ == '__main__':
    app.run(debug=True)
