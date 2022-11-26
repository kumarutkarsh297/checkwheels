from django.shortcuts import render
import os
import numpy as np
import pandas as pd
from sklearn import metrics , model_selection
from sklearn.naive_bayes import GaussianNB

# Create your views here.

def convert(a,b):
    d={}
    a=list(a)
    for i in b:
        d[i]=a.index(i)
    return d
def update(a,b):
    a=list(a)
    for k in b:
        b[k] = a[b[k]]
    return b

def index(request):
    return render(request, 'index.html')
def result(request):
    data = pd.read_csv('static/cars.csv')
    buying = convert(data['buying'],data['buying'].unique())
    maintenance = convert(data['maintenance'],data['maintenance'].unique())
    doors = convert(data['doors'],data['doors'].unique())
    persons = convert(data['persons'],data['persons'].unique())
    lug_boot = convert(data['lug boot'],data['lug boot'].unique())
    safety = convert(data['safety'],data['safety'].unique())
    acceptability = convert(data['acceptability'],data['acceptability'].unique())

    data['buying'],_ = pd.factorize(data['buying'])
    data['maintenance'],_ = pd.factorize(data['maintenance'])
    data['doors'],_ = pd.factorize(data['doors'])
    data['persons'],_ = pd.factorize(data['persons'])
    data['lug boot'],_ = pd.factorize(data['lug boot'])
    data['safety'],_ = pd.factorize(data['safety'])
    data['acceptability'],class_names = pd.factorize(data['acceptability'])

    buying = update(data['buying'],buying)
    maintenance = update(data['maintenance'],maintenance)
    doors = update(data['doors'],doors)
    persons = update(data['persons'],persons)
    lug_boot = update(data['lug boot'],lug_boot)
    safety = update(data['safety'],safety)
    acceptability = update(data['acceptability'],acceptability)

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=123)
    model = GaussianNB()
    model.fit(X_train, y_train)
    b = request.POST.get('b')
    m = request.POST.get('m')
    d = request.POST.get('d')
    p = request.POST.get('p')
    l = request.POST.get('l')
    s = request.POST.get('s')

    b = buying[b]
    m = maintenance[m]
    d = doors[d]
    p = persons[p]
    l = lug_boot[l]
    s = safety[s]

    res = model.predict([[b,m,d,p,l,s]])[0]
    res1_k = list(acceptability.keys())
    res1_v = list(acceptability.values())
    context = {"result":(res1_k[res1_v.index(res)])}
    return render(request, 'result.html', context)