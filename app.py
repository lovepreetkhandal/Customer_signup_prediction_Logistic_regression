

import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__,  template_folder='Templates')
## Load the model
clfmodel=pickle.load(open('logistics.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=clfmodel.predict(final_input)
    

    
    
    if output == 0:
        output= "The customer will not singup for the mail"
    elif output ==1:
        output= "The customer will singup for the mail"

    print(output)
    
    return output
    


if __name__=="__main__":
    app.run(debug=True)

