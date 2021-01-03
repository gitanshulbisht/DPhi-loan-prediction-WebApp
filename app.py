#import libraries
#import pickle
import numpy as np
from flask import Flask, request, render_template,flash
from flask_cors import CORS, cross_origin
import pandas as pd
import joblib
import os


#app name
app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['SECRET_KEY'] = os.urandom(24)
model = joblib.load('loan_model.pkl')


#home page
@app.route('/')
@cross_origin(supports_credentials=True)
def home():
    return render_template('index.html')

#predict the result and return it
@app.route('/predict',methods=['POST'])
@cross_origin(supports_credentials=True)
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print("In predict func")
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Gender_Male',
            'Married_Yes', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Not_Graduate', 'Self_Employed_Yes','Property_Area_Semiurban','Property_Area_Urban']

    features = [x for x in request.form.values()]
    print("features :{}".format(features))
    applicantIncomeFT=int(features[0])
    coapplicantIncomeFT=int(features[1])
    loanAmountFT=int(features[2])
    loanAmountTermFT=int(features[3])
    creditHistoryFT = int(features[4])

    #handeling categorical variables
    gender=features[5]
    if gender=="male":
        genderMaleFT=1
    else:
        genderMaleFT=0

    married=features[6]
    if married=="Yes":
        marriedYesFT=1
    else:
        marriedYesFT=0

    dependent=features[7]
    if dependent=="0":
        dependent1FT=0
        dependent2FT=0
        dependent3FT=0
    elif dependent=="1":
        dependent1FT = 1
        dependent2FT = 0
        dependent3FT = 0
    elif dependent == "2":
        dependent1FT = 0
        dependent2FT = 1
        dependent3FT = 0
    else:
        dependent1FT = 0
        dependent2FT = 0
        dependent3FT = 1

    education=features[8]
    if education=="notGraduate":
        educationNotGraduateFT=1
    else:
        educationNotGraduateFT=0
    selfEmployed=features[9]
    if selfEmployed=="Yes":
        selfEmployedYesFT=1
    else:
        selfEmployedYesFT=0

    propertyArea=features[10]
    if propertyArea=="semiurban":
        semiUrbanFT=1
        urbanFT=0
    elif propertyArea=="urban":
        semiUrbanFT=0
        urbanFT=1
    else:
        semiUrbanFT=0
        urbanFT=0
    oneHotFeatures=[applicantIncomeFT,coapplicantIncomeFT,loanAmountFT,loanAmountTermFT,creditHistoryFT,genderMaleFT,marriedYesFT,dependent1FT,dependent2FT,dependent3FT,educationNotGraduateFT,selfEmployedYesFT,semiUrbanFT,urbanFT]
    values = [np.array(oneHotFeatures)]
    print("values :{}". format(values))
    df_web=pd.DataFrame(values,columns=cols)
    print("df_web :{}".format(df_web))
    prediction = model.predict(df_web)
    print("prediction :{}".format(prediction))
    result = prediction[0]
    print("result :{}".format(result))
    if(result==0):
        result_text="Rejected"
        category = 'danger'
    else:
        result_text="Accepted"
        category = 'success'
    flash(result_text, category=category)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)