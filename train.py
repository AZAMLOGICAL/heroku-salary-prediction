import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv(r'C:\Users\HP\Desktop\DATA ANALYST\datasets\hiring.csv')
print(dataset)

dataset.rename(columns={'test_score(out of 10)':'test_score','interview_score(out of 10)':'interview_score'},inplace=True)

dataset.experience.fillna(0,inplace=True)
dataset.test_score.fillna(dataset.test_score.mean(),inplace=True)

X=dataset.iloc[:,:3]
def convert_to_int(word):
    word_dic={0:0,'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11}
    return word_dic[word]

X['experience']=X.experience.apply(lambda x:convert_to_int(x))

y=dataset.iloc[:,-1]

regressor=LinearRegression()
regressor.fit(X,y)
print('Model training is done')

joblib.dump(regressor,'hiring_model.pkl')

print(regressor.predict([[1,8,9]]))
