import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
from flask_cors import cross_origin
import numpy as np
import pandas as pd

application = Flask(__name__) # initializing a flask app
app=application
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/', methods=['GET'])
@cross_origin()
def homePage():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
@cross_origin()
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("index.html",prediction_text="The House price prediction is {}".format(output))



if __name__ == "__main__":
   # app.run(debug=True)
    app.run(host= "127.0.0.1")
   
