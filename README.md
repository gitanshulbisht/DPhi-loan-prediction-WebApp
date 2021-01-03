# Loan Request Predction

A Web App created for binary classification of Loan request Accepted Or Rejected

# Form Input Data Description
Loan_ID: A unique ID assigned to every loan applicant
Gender: Gender of the applicant (Male, Female)
Married: The marital status of the applicant (Yes, No)
Dependents: No. of people dependent on the applicant (0,1,2,3+)
Education: Education level of the applicant (Graduated, Not Graduated)
Self_Employed: If the applicant is self-employed or not (Yes, No)
ApplicantIncome: The amount of income the applicant earns
CoapplicantIncome: The amount of income the co-applicant earns
LoanAmount: The amount of loan the applicant has requested for
Loan_Amount_Term: The  no. of days over which the loan will be paid
Credit_History: A record of a borrower's responsible repayment of debts (1- has all debts paid, 0- not paid)
Property_Area : The type of location where the applicant’s property lies (Rural, Semiurban, Urban)
Loan_Status: Loan granted or not (1 for granted, 0 for not granted)[Predicted using Logistic regression model]

# Model Creation
A basic logistic regression model is used for the above binary classification.

# Deployment
Flask is the frame work used for deployment app .
Pickeled model is used for the prediction and the result is then returned to the fornt-end.

# UI
HTML,CSS,BootStrap,Jinja Templates
