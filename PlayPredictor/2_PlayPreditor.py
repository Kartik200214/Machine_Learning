import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def MPlayPreditor(data_path):

    data = pd.read_csv(data_path,index_col=0)

    print("Size of Actual dataset",len(data))

    feature_names = ['Whether','Temperature']

    print("Name of Features", feature_names)

    X = data[feature_names]

    y = data.Play

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)

    print("Size of Training dataset",len(X_train))

    print("Size of Testing dataset",len(X_test))


def main():
    MPlayPreditor("PlayPredictor.csv")

if __name__ == "__main__":
    main()