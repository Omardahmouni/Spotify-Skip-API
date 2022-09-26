import numpy as np  #for numerical python
import matplotlib.pyplot as plt         #for 2D Graphics
import pandas as pd         #for dataset
import seaborn as sb        #for more graphical properties

"""let's read the file and take an overview on it"""

#let's load the data that I got after applying features engineering
final_data=pd.read_csv("final_data.csv")

#setting the index columns for our model
final_data.set_index("session_id",inplace=True)

#let's import the main element to choose the best k features
from sklearn.feature_selection import SelectKBest,mutual_info_classif
#Features vector
X = final_data.drop("skipped",axis=1)
#selecting the target vector
Y=final_data["skipped"]
#let's select the best 15 features for our model
X_new=SelectKBest(mutual_info_classif,k=15).fit(X,Y)
best_features=list(X_new.get_feature_names_out())
columns_to_drop=[column for column in X.columns if column not in best_features]
X.drop(columns_to_drop,axis=1,inplace=True)
#the new features
print("the features to work with:\n"+str(list(X.columns)))

"""After chosing the best features let's build our model

THE GRADIENT BOOSTING CLASSIFIER
"""

#let's start by importing the main library for xgboost classifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#let's split our data into train and test with 25% of rows as test rows and 75% of trained data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state=111)

#now let's create our model with the default params
XGB_Classifier=XGBClassifier()

#then let's fit and  train it
XGB_Classifier.fit(x_train,y_train)

print("\nthis model accuracy is: "+str(XGB_Classifier.score(x_test,y_test)*100)[:5],"%")

#let's import the pickle model to create the pickle file
import pickle
#let's generate our pickle model
pickle.dump(XGB_Classifier,open("XGB_model.pkl",'wb'))
print(str(X.iloc[0,:]))
print("\n"+str(Y[0]))