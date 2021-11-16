from flask import Flask, request
import pickle
import json
import numpy as np


app = Flask(__name__)

classifier = pickle.load(open('classifier.pickle','rb'))
scalar = pickle.load(open('sc.pickle','rb'))

@app.route("/predict",methods=["POST"])
def predict():
    request_data = request.get_json(force=True)
    # print(request_data)
    age = request_data['age']
    price = request_data['price']
    pred = classifier.predict(scalar.transform(np.array([[int(age),int(price)]])))
    pred_prob = classifier.predict_proba(scalar.transform(np.array([[age,price]])))[:,1]
    # print(pred)
    if pred == 0:
        return f"The user will Not Buy this item, with probability{pred_prob}"
    else:
        return f"The user will Buy this item, with probability{pred_prob}"


if __name__ == '__main__':
   app.run(host='0.0.0.0',port=8000, debug = True)