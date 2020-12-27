#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

#app name
app = Flask(__name__)

#load the saved model
def load_model():
    return pickle.load(open('loan_model.pkl', 'rb'))

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
    values = [np.array(features)]
    model = load_model()
    prediction = model.predict(values)
    result = labels[prediction[0]]
    print(features)
    return render_template('index.html', output='The loan is {}'.format(result))


if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)