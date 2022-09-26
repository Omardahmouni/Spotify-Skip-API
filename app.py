from flask import Flask , jsonify,render_template,request
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV



app=Flask(__name__)

def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 15)
	xgboost_model=pickle.load(open("XGB_model.pkl",'rb'))
	result =xgboost_model.predict(to_predict)
	return result[0]


@app.route('/',methods=['GET'])
def helloworld():
    return render_template('index.html')

@app.route('/', methods= ['POST'])

def predict():
	if request.method=='POST':
		to_predict_list=request.form.to_dict()
		to_predict_list=list(to_predict_list.values())
		to_predict_list=list(map(float, to_predict_list))
		result=ValuePredictor(to_predict_list)       
		if int(result)==1:
			prediction=' the music will be skipped '
		else:
			prediction=" the music won't be skipped "          
		return render_template("index.html", prediction = prediction)


if __name__=='__main__':
    app.run(port=3000,debug=True)
    
   
