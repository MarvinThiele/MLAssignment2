# This file will implement a KNN (k-nearest-neighbor algorithm)

class kNearestNeighborsClassifier():
    dataframe = []
    k = 5

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




