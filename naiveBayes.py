from categoryEncoder import categoryEncoder
import copy

class naivesBayesClassifier:
    name = "Naive Bayes"

    def __init__(self):
        pass

    def classifyInstance(self, instance):
        raise NotImplementedError

    def classifyData(self, trainingData, testData):
        encoder = categoryEncoder()

        # Transform the real-valued data into categorical data to be able to apply naive bayes
        trainingData = encoder.fitTransform(trainingData)
        testData = encoder.transform(testData)

        # Calculate the Probabilties for the negative and posivie Class
        pPositive = 0
        pNegative = 0
        for instance in trainingData:
            if instance[57] == 0:
                pNegative += 1
            else:
                pPositive += 1
        sum = pPositive + pNegative
        pPositive = pPositive / sum
        pNegative = pNegative / sum

        # Calculate the conditional probabilities
        # First calculate the count for each possible value of each feature for each class
        probabilities = []
        for i in range(0, 57):
            featureDict = {}
            numberDict = {}
            numberDict[0] = 0
            numberDict[1] = 1
            for possibleValue in range(0,6):
                featureDict[possibleValue] = copy.deepcopy(numberDict)
            for instance in trainingData:
                featureDict[instance[i]][instance[57]] += 1
            probabilities.append(featureDict)

        # Divide that count by the total amount of positive and negative samples of that feature
        for i in range(0, 57):
            sumPositve = 0
            sumNegative = 0
            for j in range(0, 6):
                sumPositve += probabilities[i][j][1]
                sumNegative += probabilities[i][j][0]
            for j in range(0, 6):
                probabilities[i][j][1] = probabilities[i][j][1] / sumPositve
                probabilities[i][j][0] = probabilities[i][j][0] / sumNegative

        # Predict each instance of the test dataset by multiplying the conditional probabilities calculated above
        prediction = []
        for instance in testData:
            productPositive = pPositive
            productNegative = pNegative
            for i in range(0, 57):
                productPositive = productPositive * probabilities[i][instance[i]][1]
                productNegative = productNegative * probabilities[i][instance[i]][0]
            if productPositive > productNegative:
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction

