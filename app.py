from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model/regmodel.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final = scaler.transform([features])
    output = model.predict(final)[0]
    return render_template("index.html", prediction_text=f"Predicted Price: ${output:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
