from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('profanity_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

@app.route('/check', methods=['POST'])
def check_profanity():
    data = request.get_json()
    texts = data.get('texts', [])
    texts_vect = vectorizer.transform(texts)
    predictions = model.predict(texts_vect)
    probabilities = model.predict_proba(texts_vect)[:, 1]

    results = [{'text': text, 'is_profane': bool(pred), 'probability': prob} for text, pred, prob in zip(texts, predictions, probabilities)]
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
