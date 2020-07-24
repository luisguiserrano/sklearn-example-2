import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import json

def build_dataset():
    dataset = pd.DataFrame({'x_1':[0,0,1,1,2,2],
                        'x_2':[0,1,0,2,1,2],
                        'y':[0,0,0,1,1,1]})
    result = {}
    result["dataset"] = dataset
    return result

def preprocess_data(dataset):
    features = np.array(dataset[['x_1', 'x_2']])
    labels = np.array(dataset['y'])
    result = {}
    result["features"] = features
    result["labels"] = labels
    return result

def return_model(model_name = "logistic_regression"):
    if model_name == 'logistic_regression':
        model = LogisticRegression()
    if model_name == 'decision_tree':
        model = DecisionTreeClassifier()
    if model_name == 'svm':
        model = SVC()
    return model

def train_model(model, features, labels):
    model.fit(features, labels)
    return model

def score_model(model, features, labels):
    score = model.score(features, labels)
    return score

def predict(model, features):
    predictions = model.predict(features)
    predictions = np.array(predictions)
    return json.dumps(predictions, cls=NumpyArrayEncoder)

#dataset = build_dataset()
#features, labels = preprocess_data(dataset)
#model = return_model("decision_tree")
#model = train_model(model, features, labels)
#score = score_model(model, features, labels)
#predictions = predict(model, features)

#print("Score:", score)
#print("Predictions:", predictions)