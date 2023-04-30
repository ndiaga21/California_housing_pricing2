import pickle
from flask import Flask, request,app, jsonify, url_for, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)
# load the model
catmodel = pickle.load(open('catmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = catmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    new_data= scalar.transform(np.array(data).reshape(1,-1))
    print(new_data)
    output = catmodel.predict(new_data)[0]
    return render_template('home.html', prediction_text="The house price prediction is {}".format(output))
if __name__== "__main__":
    app.run(debug=True)