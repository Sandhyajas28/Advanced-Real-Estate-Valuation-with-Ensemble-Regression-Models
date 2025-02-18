from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("real_estate_model .pkl", "rb"))

# Route for Home Page
@app.route('/')
def home():
    return render_template('home.html')  #Links to home.html

# Route for Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input from the form
            transaction_date = float(request.form['transaction_date'])
            house_age = float(request.form['house_age'])
            distance_to_MRT = float(request.form['distance_to_MRT'])
            convenience_stores = float(request.form['convenience_stores'])
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])

            # Create input array
            input_data = np.array([[transaction_date, house_age, distance_to_MRT, convenience_stores, latitude, longitude]])

            # Make prediction
            predicted_price = model.predict(input_data)[0]

            return render_template('predict.html', prediction=predicted_price)

        except Exception as e:
            return render_template('predict.html', error=str(e))

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
