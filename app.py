from os import defpath
from flask import Flask, render_template,request,jsonify, redirect, url_for
import pickle
import xgboost as xgb
from flask.json import load
import numpy as np


app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape((1,-1))
    prediction = model.predict(final_features)

    output =prediction
    
   return render_template('index.html', prediction_text='House price is $ {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
