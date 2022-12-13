import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def PlayPreditor(data_path):

    #Step 1 :Load data
    data = pd.read_csv(data_path,index_col=0)

    print("Size of Actual dataset",len(data))

    # Step 2: Clean , Prepare and manipulate data
    feature_names = ['Whether','Temperature']

    print("Names of Features",feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    #creating lableEncoder
    le = preprocessing.LabelEncoder()

    #Convering string lables into numbers
    weather_encoded = le.fit_transform(whether)
    print(weather_encoded)

    #Convering string lables into numbers
    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)

    print(temp_encoded)

    #combinig weather and temp into single listof tuples
    features = list(zip(weather_encoded,temp_encoded))

    #Step 3 : Train Data
    model = KNeighborsClassifier(n_neighbors=3)
    
    #Train the model using the training sets
    model.fit(features,label)

    #predit Output
    predicted = model.predict([[0,2]]) #0: overcast , 2: Mild
    print(predicted)

    if predicted:
        print("You can play")
    else:
        print("No Play")

def main():
    print("Machine Learning Application")

    print("Play preditor application using K Nearest Knighbor algorithm")

    PlayPreditor("PlayPredictor.csv")


if __name__ == "__main__":
    main()