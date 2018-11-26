from sklearn import tree
from categoryEncoder import categoryEncoder

class decisionTreeClassifier():
    name = "Decision Tree"

    def __init__(self):
        pass

    def classifyData(self, trainingData, testData):
        encoder = categoryEncoder()

        # Transform the real-valued data into categorical data to be able to apply naive bayes
        trainingData = encoder.fitTransform(trainingData)
        testData = encoder.transform(testData)

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

        clf = tree.DecisionTreeClassifier()
        clf.fit(onlyFeaturesTrain, onlyClassTrain)
        return clf.predict(onlyFeaturesTest)