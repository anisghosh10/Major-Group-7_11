import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle as pkl

app = Flask(__name__)
cont_dict = {0:'canada',1:'china',2:'india',3:'japan',4:'switzerland',5:'uk'}
# currencies = {0:"CAD",1:'Yuan',2:'Rupee',3:'Japanese Yen',4:'Swiss frand',5:'Pound'}
currency = {0:"Can$",1:'¥',2:'₹',3:'JP¥',4:'₣',5:'£'}
df = pd.read_csv('weights_15.csv')
# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    '''
    For rendering results on HTML GUI
    '''
    
    input_features = [float(x) for x in request.form.values()]
    year = int(input_features[0])

    from_contry = int(input_features[1])
    fac_1 = pd.read_csv(f'Data/eco_{cont_dict[from_contry]}.csv') # load data for from country
    data1 = pd.concat([df,fac_1],axis=1)
    data1 = data1.iloc[year-1981,:].values # fetch data for a particular year
    model1 = pkl.load(open(f'Models/model_{cont_dict[from_contry]}.pkl','rb')) # prediction model
    scaler1 = pkl.load(open(f'Scalers/scaler_{cont_dict[from_contry]}.pkl','rb'))
    data1 = scaler1.transform([data1]) # scaling
    pred_1 = model1.predict(data1)[0] # predicting

    
    to_contry = int(input_features[2])
    fac_2 = pd.read_csv(f'Data/eco_{cont_dict[to_contry]}.csv') # load data for from country
    data2 = pd.concat([df,fac_2],axis=1)
    data2 = data2.iloc[year-1982,:].values   
    model2 = pkl.load(open(f'Models/model_{cont_dict[to_contry]}.pkl','rb')) # prediction model
    scaler2 = pkl.load(open(f'Scalers/scaler_{cont_dict[to_contry]}.pkl','rb'))
    data2 = scaler2.transform([data2]) # scaling
    pred_2 = model2.predict(data2)[0] # predicting
    
    if from_contry == to_contry:
        exc_rate = 1
    else:
        exc_rate = np.round((pred_2/pred_1),2) # currency exchange rate
    amt = int(input_features[-1])
    
    return render_template('index.html', prediction_text=f'{amt} {currency[from_contry]} = {amt*exc_rate} {currency[to_contry]}',color_text='red')


if __name__ == "__main__":
    app.run(debug=True)
