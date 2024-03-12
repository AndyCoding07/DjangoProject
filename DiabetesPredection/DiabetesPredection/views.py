from django.shortcuts import render
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def index(request):
    return render(request, "index.html")

def predictpage(request):
    return render(request, "predictpage.html")

def result(request):
    dataframe = pd.read_csv("DiabetesPredection\diabetes.csv")

    X = dataframe.drop("Outcome", axis=1)
    Y = dataframe["Outcome"]

    model = DecisionTreeClassifier(random_state = 0)
    model.fit(X, Y)

    preg = float(request.GET['preg'])
    glucose = float(request.GET['glucose'])
    bp = float(request.GET['bp'])
    skin = float(request.GET['skin'])
    insulin = float(request.GET['insulin'])
    bmi = float(request.GET['bmi'])
    dpf = float(request.GET['dpf'])
    age = float(request.GET['age'])

    # predicting outcome based on users input 
    prediction = model.predict([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    if (prediction == [1]):
        result = "POSITIVE for Diabetes."
        
    else:
        result = "NEGATIVE for Diabetes."

    return render(request, "predictpage.html", {"result": result})