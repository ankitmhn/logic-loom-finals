from flask import Flask, jsonify, request
from model import classify
import pickle

model = pickle.load(open('finalized_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Flask Web Server!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    # Dummy prediction for example purposes
    # prediction = "Human" if len(text) % 2 == 0 else "AI"

    # prediction = classify([text])
    prediction = model.predict(text)
    return jsonify({'prediction': prediction})


if __name__ == '_main_':
    app.run(debug=True)
