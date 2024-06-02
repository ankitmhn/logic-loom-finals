import os
from flask import Flask, Response, jsonify, request, redirect
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

train = pd.read_csv("train_set.csv")
X = train['Article_content']
y = train['Article_type']
Tfidf = TfidfVectorizer(encoding='utf-8')
X_vectorized = pd.DataFrame.sparse.from_spmatrix(
    Tfidf.fit_transform(X), columns=Tfidf.get_feature_names_out())
y[y == 'Human-written'] = 0
y[y == 'AI-generated'] = 1
y_encoded = y.astype('int64')
S = SMOTE(random_state=42)
X_res, y_res = S.fit_resample(X_vectorized.to_numpy(), y_encoded.to_numpy())
XGB = XGBClassifier(random_state=42, eta=0.85, max_depth=200)
XGB.fit(X_res, y_res)


def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)


@app.route('/', methods=['GET'])
def home():  # pragma: no cover
    content = get_file('index.html')
    return RenderTemplate(content, mimetype="text/html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    text_vectorized = pd.DataFrame.sparse.from_spmatrix(
        Tfidf.transform([text]), columns=Tfidf.get_feature_names_out())
    predict = XGB.predict(text_vectorized)
    return str(predict[0])
    # return jsonify({'prediction': predict})


if __name__ == '_main_':
    app.run(debug=True)
