from flask import Flask,render_template,request
import joblib
import numpy as np

app=Flask(__name__)

model=joblib.load('hiring_model.pkl')

@app.route('/')
def hello_world():
    return render_template('base.html')

@app.route('/predict',methods=['POST'])
def predict():

    exp=request.form.get('experience')
    score=request.form.get('test_score')
    interview_score=request.form.get('interview_score')
    
    result=model.predict(np.array([[int(exp),int(score),int(interview_score)]]))[0]
    output=round(result,2)

    return render_template('base.html',prediction_text=f'The salary is around {output}')

@app.route('/feedback')
def feedback():
    return 'Welcome to the feedback page'

@app.route('/help')
def help():
    return 'Welcome to the help page'


app.run(debug=True) 