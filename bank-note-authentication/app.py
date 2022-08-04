from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

pickle_in = open('classifier.pkl','rb')
rf = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome all"

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')

    prediction = rf.predict([[variance,skewness,curtosis,entropy]])

    return f"The predicted value is {str(prediction)}"




@app.route('/predict_file',methods = ['post'])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    # variance = request.args.get('variance')
    # skewness = request.args.get('skewness')
    # curtosis = request.args.get('curtosis')
    # entropy = request.args.get('entropy')
    prediction = rf.predict(df_test)
    return f"The predicted value is {list(prediction)}"















if __name__ == '__main__':
    app.run()

