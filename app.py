#import libraries
#import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib


#app name
app = Flask(__name__)
model = joblib.load('loan_model.pkl')


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
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Gender_Male',
            'Married_Yes', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Not_Graduate', 'Self_Employed_Yes','Property_Area_Semiurban','Property_Area_Urban']

    print( request.form.values())
    features = [int(x) for x in request.form.values()]
    print("features :{}".format(features))
    values = [np.array(features)]
    print("values :{}". format(values))
    df_web=pd.DataFrame(values,columns=cols)
    print("df_web :{}".format(df_web))
    prediction = model.predict(df_web)
    print("prediction :{}".format(prediction))
    result = prediction[0]
    print("result :{}".format(result))
    if(result==0):
        result_text="Rejected"
    else:
        result_text="Accepted"
    return render_template('index.html', output='The loan request is {}'.format(result_text))


if __name__ == "__main__":
    app.run(debug=True)