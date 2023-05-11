#from crypt import methods
import traceback
from flask import Flask, jsonify, render_template
from requests import request
from flask import request
import pickle
import numpy as np
import pandas as pd

#load pickle model
model = pickle.load(open('model1.pkl','rb'))

#create flask app
app = Flask(__name__)

@app.route('/')
def home():
    #return "Welcome to Diabetic predictions"
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    tlc = request.form.get('tlc')
    lym=request.form.get('lym')
    poly=request.form.get('poly')
    mono=request.form.get('mono')
    eosino=request.form.get('eosino')
    protein=request.form.get('protein')
    sugar=request.form.get('sugar')
    #input_query=np.array([[tlc,lym,poly,mono,eosino,protein,sugar]])
    input_query=np.array([[tlc,protein,sugar]])
    result =model.predict(input_query)[0]
   #result ={'tlc':tlc,'lym':lym,'poly':poly,'mono':mono,'eosino':eosino,'protein':protein,'sugar':sugar,'Age':Age}
    # --> correct return jsonify({'person is diabatic':str(result)})
    if result:
        try:
            return jsonify({'Person is: ':str(result)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')
            

if __name__=='__main__':
    app.run(debug=True)