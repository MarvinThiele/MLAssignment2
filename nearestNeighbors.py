# This file will implement a KNN (k-nearest-neighbor algorithm)
from sklearn import neighbors
import time

class kNearestNeighborsClassifier():
    dataframe = []
    k = 5
    name = "K-Nearest-Neighbor"

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def classifyInstance(self, instance):
        euclideanDistances = []
        for i, dataPoint in enumerate(self.dataframe):
            difference = 0
            for i in range(0, len(instance)-1):
                difference += abs(dataPoint[i] - instance[i])
            euclideanDistances.append((i, difference))
        sortedArray = sorted(euclideanDistances, key=lambda x: x[1])

        positive = 0
        negative = 0
        for i in range(0, self.k):
            if self.dataframe[sortedArray[i][0]][57] == 1:
                positive += 1
            else:
                negative += 1
        if positive > negative:
            return 1
        else:
            return 0

    def classifyDataSelfImplemented(self, trainingData, testData):
        self.dataframe = trainingData

        results = []
        for i, instance in enumerate(testData):
            results.append(self.classifyInstance(instance))
            print(i)

        return results

    def classifyData(self, trainingData, testData, returnTrainingTime=False):
        onlyFeaturesTrain = []
        onlyClassTrain = []
        for instance in trainingData:
            onlyFeaturesTrain.append(instance[0:57])
            onlyClassTrain.append(int(instance[57]))

        onlyFeaturesTest = []
        onlyClassTest = []
        for instance in testData:
            onlyFeaturesTest.append(instance[0:57])
            onlyClassTest.append(int(instance[57]))

        if returnTrainingTime:
            timeStart = time.time()

        clf = neighbors.KNeighborsClassifier()
        clf.fit(onlyFeaturesTrain, onlyClassTrain)

        if returnTrainingTime:
            timeEnd = time.time()
            return (timeEnd-timeStart)*1000

        return clf.predict(onlyFeaturesTest)




