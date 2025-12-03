
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"

# Load model
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
feature_names = model_data["features"]


@app.route('/')
def home():
    return render_template('index.html', features=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = []

        for feature in feature_names:
            val = float(request.form.get(feature))
            values.append(val)

        values = np.array(values).reshape(1, -1)
        prediction = model.predict(values)[0]

        return render_template("index.html", result=prediction, features=feature_names)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}", features=feature_names)


if __name__ == '__main__':
    app.run(debug=True)