from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class KNeighborsClassifierX:
    def fit(self,trainingdata,trainingtarget):
        self.TrainingData = trainingdata
        self.Trainingtarget = trainingtarget

    def closest(self,row):
        minimumdistance = euc(row,self.TrainingData[0])
        minimumindex = 0

        for i in range(1,len(self.TrainingData)):
            Distance = euc(row,self.TrainingData[i])
            minimumindex = 0
            if Distance < minimumdistance:
                minimumdistance = Distance
                minimumindex = i 
        return self.Trainingtarget[minimumindex]

    def predict(self,TestData):
        predictions = []
        for value in TestData:
            result = self.closest(value)
            predictions.append(result)
        return predictions


def MarvellousML():
    Dataset = load_iris()       # 1 Load the data

    Data = Dataset.data
    Target = Dataset.target

    # 2 : Manipulate the data
    Data_train, Data_test, Target_train, Target_test = train_test_split(Data, Target, test_size = 0.5)

    Classifier = KNeighborsClassifierX()

    # 3 : Build the model
    Classifier.fit(Data_train, Target_train)

    # 4 : Test the model
    Predictions = Classifier.predict(Data_test)

    
    Accuracy = accuracy_score(Target_test, Predictions) # check Accuracy

    # 5: Improve  ---Missing

    return Accuracy

def main():
    Ret = MarvellousML()

    print("Acuracy of Iris dataset with KNN is ",Ret * 100)

if __name__ == "__main__":
    main()