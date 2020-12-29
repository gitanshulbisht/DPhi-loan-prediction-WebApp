#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

#app name
app = Flask(__name__)
model = pickle.load(open('loan_model.pkl', 'rb'))



#home page
@app.route('/')
def home():
    return render_template('index.html')

#predict the result and return it
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print("In predict func")
    labels = ['0', '1']

    print( request.form.values())
    features = [str(x) for x in request.form.values()]
    print("features :{}".format(features))
    values = [np.array(features)]
    print("values :{}". format(values))

    print("model :{}".format(model))
    prediction = model.predict(values)
    print("prediction :{}".format(prediction))
    result = labels[prediction[0]]
    print("result :{}".format(result))
    if(result==0):
        result_text="Rejected"
    else:
        result_text="Accepted"
    return render_template('index.html', output='The loan request is {}'.format(result_text))


if __name__ == "__main__":
    app.run(debug=True)