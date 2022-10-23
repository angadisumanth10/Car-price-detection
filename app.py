from flask import Flask, render_template,url_for,request

import pickle

import numpy as np
#initialising flask by giving name
app = Flask(__name__)


model = pickle.load(open('car.pkl','rb'))
@app.route("/")
def home():
    return render_template('car.html') 


@app.route("/predict", methods=["POST"])
def predict():
    float_features =[float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    model = pickle.load(open('car.pkl','rb'))
    pred = model.predict(features)
    return render_template("car.html", text_prediction ="Your Model price may be {} ".format(pred)) 

if __name__=='__main__':
    app.run()