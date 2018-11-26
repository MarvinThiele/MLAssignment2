from nearestNeighbors import kNearestNeighborsClassifier
from naiveBayes import naivesBayesClassifier
from decisionTree import decisionTreeClassifier
import random
import copy
import math

def main():
    df = []
    with open("spambase.csv") as file:
        for line in file:
            line = line.split(",")
            for i, number in enumerate(line):
                line[i] = float(number)
            df.append(line)

    information = {}
    with open("spambase_description.txt") as file:
        for i, line in enumerate(file):
            line_info = {}
            line = line.split(" ")
            line_info['min'] = float(line[1])
            line_info['max'] = float(line[2])
            line_info['avg'] = float(line[3])
            line_info['stddev'] = float(line[4])
            line_info['coeffvar'] = float(line[5])
            information[i] = line_info



    random.shuffle(df)
    border = int(len(df)*0.7)
    trainingData = df[0 : border]
    testData = df[border:]

    nb = naivesBayesClassifier()
    dt = decisionTreeClassifier()
    knn = kNearestNeighborsClassifier(df)

    #dt.classifyData(trainingData, testData)
    crossValidation(df, [nb, dt, knn])

    #crossValidation(df, nb)
    #crossValidation(df, knn)

    #predicted = knn.classifyData(trainingData, testData)
    #reportStatistics(testData, predicted)

def reportStatistics(testData, prediction):
    correct = 0
    false = 0
    for i in range (0, len(testData)):
        if testData[i][57] == prediction[i]:
            correct += 1
        else:
            false += 1
    print("Accuracy: " + str((correct/len(testData))*100))
    return round(correct/len(testData), 4)

def crossValidation(df, classifiers):
    random.shuffle(df)

    # Create the 10 data packets
    dataParts = []
    for i in range(0, 10):
        if i == 0:
            dataParts.append(df[0:int(len(df) * 0.1)])
        else:
            dataParts.append(df[int(len(df) * (i/10)):int(len(df) * (i+1)/10)])

    # Test the classifier 10 times
    results = [[], [], []]
    for i in range(0, 10):
        testData = dataParts[i]
        trainingData = []
        for j in range(0, 10):
            if j != i:
                trainingData.extend(dataParts[j])
            else:
                continue

        for j, classifier in enumerate(classifiers):
            results[j].append(reportStatistics(testData, classifier.classifyData(trainingData, testData)))

    # Average the results
    sums = [0,0,0]
    averages = [0,0,0]
    averagesStrings = ["", "", ""]
    for i in range(0, 10):
        for j in range(0, len(results)):
            sums[j] += results[j][i]
    for i, sum in enumerate(sums):
        averages[i] = sum / 10
        averagesStrings[i] = str(round(averages[i], 4))
    for i, avgstring in enumerate(averagesStrings):
        while len(avgstring) < 6:
            avgstring += "0"
        averagesStrings[i] = avgstring

    # Standard Deviations
    stdMean = copy.deepcopy(results)
    for i in range(0, 10):
        for j in range(0, len(results)):
            stdMean[j][i] = pow(stdMean[j][i] - averages[j], 2)
    sums = [0, 0, 0]
    for i in range(0, 10):
        for j in range(0, len(results)):
            sums[j] += stdMean[j][i]
    std = [0,0,0]
    stdStrings = ["", "", ""]
    for i, sum in enumerate(sums):
        std[i] = math.sqrt(sum / 10)
        stdStrings[i] = str(round(std[i], 4))
    for i, stdstring in enumerate(stdStrings):
        while len(stdstring) < 6:
            stdstring += "0"
            stdStrings[i] = stdstring

    print("")
    print("------------------------------------------------------------")
    print(" Fold    Naive Bayes     Decision Tree    K-Nearest-Neighbor")
    print("------------------------------------------------------------")
    for i in range(0, 10):
        line = "   "+str(i+1)
        for j in range(0, len(results)):
            number = str(results[j][i])
            while len(number) < 6:
                number += "0"
            if i == 9 and j == 0:
                line += "        "+number
            else:
                line += "         " + number
        print(line)
    print("------------------------------------------------------------")
    print(" avg         " + str(averagesStrings[0])+"         " + str(averagesStrings[1]) + "         " + str(averagesStrings[2]))
    print(" stdev       " + str(stdStrings[0]) + "         " + str(stdStrings[1]) + "         " + str(stdStrings[2]))
    print("------------------------------------------------------------")

if __name__ == "__main__":
    main()