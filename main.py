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

    nb = naivesBayesClassifier()
    dt = decisionTreeClassifier()
    knn = kNearestNeighborsClassifier(df)

    classifiers = [nb, dt, knn]

    reportMeanStdDev(df, [nb, dt, knn], "accuracy")
    reportMeanStdDev(df, [nb, dt, knn], "f1")
    reportMeanStdDev(df, [nb, dt, knn], "trainingTime")

    friedmanTest(df, classifiers, "accuracy")
    friedmanTest(df, classifiers, "f1")
    friedmanTest(df, classifiers, "trainingTime")

def reportAccuracy(trainingData, testData, classifier):
    prediction = classifier.classifyData(trainingData, testData)
    correct = 0
    false = 0
    for i in range (0, len(testData)):
        if testData[i][57] == prediction[i]:
            correct += 1
        else:
            false += 1
    return round(correct/len(testData), 4)

def reportF1(trainingData, testData, classifier):
    prediction = classifier.classifyData(trainingData, testData)
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    for i in range (0, len(testData)):
        if testData[i][57] == prediction[i]:
            if testData[i][57] == 1:
                truePositive += 1
            else:
                trueNegative += 1
        else:
            if testData[i][57] == 1:
                falseNegative += 1
            else:
                falsePositive += 1
    precision = truePositive / (truePositive+falsePositive)
    recall = truePositive / (truePositive+falseNegative)
    f1 = 2 * ((precision*recall)/(precision+recall))
    return round(f1, 4)

def reportTrainingTime(trainingData, testData, classifier):
    return round(classifier.classifyData(trainingData, testData, returnTrainingTime=True), 3)

def computeResults(df, classifiers, measure):
    # Create the 10 data packets
    dataParts = []
    for i in range(0, 10):
        if i == 0:
            dataParts.append(df[0:int(len(df) * 0.1)])
        else:
            dataParts.append(df[int(len(df) * (i / 10)):int(len(df) * (i + 1) / 10)])

    # Test the classifiers 10 times
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
            if measure == "f1":
                results[j].append(reportF1(trainingData, testData, classifier))
            elif measure == "accuracy":
                results[j].append(reportAccuracy(trainingData, testData, classifier))
            elif measure == "trainingTime":
                results[j].append(reportTrainingTime(trainingData, testData, classifier))
            else:
                print("Measure not found!")
                raise NotImplementedError
    return results

def reportMeanStdDev(df, classifiers, measure):
    print("")
    random.shuffle(df)
    results = computeResults(df, classifiers, measure)

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

    if measure == "f1":
        print("F1 Table:")
    elif measure == "accuracy":
        print("Accuracy Table:")
    elif measure == "trainingTime":
        print("Training Time Table:")
    else:
        raise NotImplementedError
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

def friedmanTest(df, classifiers, measure):
    print("")
    random.shuffle(df)
    results = computeResults(df, classifiers, measure)
    ranks = [[], [], []]

    for i in range(0, 10):
        if results[0][i] > results[1][i] and results[1][i] > results[2][i]:
            ranks[0].append(1)
            ranks[1].append(2)
            ranks[2].append(3)
        elif results[0][i] > results[2][i] and results[2][i]  > results[1][i]:
            ranks[0].append(1)
            ranks[1].append(3)
            ranks[2].append(2)
        elif results[1][i] > results[0][i] and results[0][i] > results[2][i]:
            ranks[0].append(2)
            ranks[1].append(1)
            ranks[2].append(3)
        elif results[1][i] > results[2][i] and results[2][i] > results[0][i]:
            ranks[0].append(3)
            ranks[1].append(1)
            ranks[2].append(2)
        elif results[2][i] > results[1][i] and results[1][i] > results[0][i]:
            ranks[0].append(3)
            ranks[1].append(2)
            ranks[2].append(1)
        elif results[2][i] > results[1][i] and results[1][i] > results[0][i]:
            ranks[0].append(3)
            ranks[1].append(2)
            ranks[2].append(1)
        else:
            # If two values are the same, we will use the normal ordering
            if results[0][i] > results[1][i]:
                ranks[0].append(1)
                ranks[1].append(2)
                ranks[2].append(3)
            elif results[1][i] > results[2][i]:
                ranks[0].append(2)
                ranks[1].append(1)
                ranks[2].append(3)
            else:
                ranks[0].append(2)
                ranks[1].append(3)
                ranks[2].append(1)
            raise NotImplementedError
        if measure == "trainingTime":
            for j in range(0,3):
                if ranks[j][i] == 1:
                    ranks[j][i] = 3
                elif ranks[j][i] == 3:
                    ranks[j][i] = 1
    # Calculate the average ranks
    sums = [0, 0, 0]
    averagesRanks = [0, 0, 0]
    averageRankStrings = ["", "", ""]
    for i in range(0, 10):
        for j in range(0, len(ranks)):
            sums[j] += ranks[j][i]
    for i, sum in enumerate(sums):
        averagesRanks[i] = sum / 10
        averageRankStrings[i] = str(round(averagesRanks[i], 1))

    nullHypothesisAverage = 2

    SSE = 0
    for average in averagesRanks:
        SSE += pow(average-nullHypothesisAverage, 2)
    friedmanTestValue = SSE * 10

    nemenyiTestValue = 2.343 * math.sqrt((3*(3+1))/(6*10))

    if measure == "f1":
        print("F1 Friedman Table:")
    elif measure == "accuracy":
        print("Accuracy Friedman Table:")
    elif measure == "trainingTime":
        print("Training Time Friedman Table:")
    else:
        raise NotImplementedError
    print("--------------------------------------------------------------------------------------------")
    print(" Dataset    Naive Bayes     Decision Tree    K-Nearest-Neighbor")
    print("--------------------------------------------------------------------------------------------")
    for i in range(0, 10):
        line = "   "+str(i+1)
        for j in range(0, len(results)):
            number = str(results[j][i])
            while len(number) < 6:
                number += "0"
            number += " ("+str(ranks[j][i])+")"
            if i == 9 and j == 0:
                line += "        "+number
            else:
                line += "         " + number
        print(line)
    print("--------------------------------------------------------------------------------------------")
    print(" avg         " + str(averageRankStrings[0])+"                " + str(averageRankStrings[1]) + "                " + str(averageRankStrings[2]))
    print("--------------------------------------------------------------------------------------------")
    print(" Friedman Test: " + str(friedmanTestValue))
    if friedmanTestValue > 6.8:
        print(" We reject the null hypothesis because the friedman value is above the critical value")
        print(" We will execute the Nemenyi Test because we rejected the null hypnothesis in the Friedman Test")
        print("--------------------------------------------------------------------------------------------")
        print(" Nemenyi Test:")
        print(" The critial value is: " + str(round(nemenyiTestValue, 4)))
        if abs(averagesRanks[0] - averagesRanks[1]) > nemenyiTestValue:
            print(" The difference between the first and the second algorithm is below the critical difference")
        else:
            print(" The difference between the first and the second algorithm is above the critical difference")
        if abs(averagesRanks[1] - averagesRanks[2]) > nemenyiTestValue:
            print(" The difference between the second and the third algorithm is below the critical difference")
        else:
            print(" The difference between the second and the third algorithm is above the critical difference")
        if abs(averagesRanks[0] - averagesRanks[2]) > nemenyiTestValue:
            print(" The difference between the first and the third algorithm is below the critical difference")
        else:
            print(" The difference between the first and the third algorithm is above the critical difference")
    else:
        print(" We confirm the null hypothesis because the friedman value is below the critical value")

    print("--------------------------------------------------------------------------------------------")





if __name__ == "__main__":
    main()