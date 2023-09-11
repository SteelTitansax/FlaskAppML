import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import plotly
import plotly.express as px
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import cufflinks as cf

if __name__ == '__main__':
    pyo.init_notebook_mode(connected=True)
    cf.go_offline()
    df = pd.read_csv('/home/titansax/Repositorio/HearthDisease/heart.csv')
    df

    X, y = df.loc[:, :'thal'], df['target']

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.3, shuffle=True)

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)

    prediction_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, prediction_knn) * 100
    print('accuracy_score score     : ', accuracy_score(y_test, prediction_knn) * 100, '%')

    # save the model to disk
    filename = 'HeartDisease_model.sav'
    joblib.dump(knn, filename)

    # load the model from disk
    # loaded_model = joblib.load(filename)
    # result = loaded_model.score(X_test, Y_test)
    # print("loaded result : "  + str(result))