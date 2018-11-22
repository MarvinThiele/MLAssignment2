from nearestNeighbors import kNearestNeighborsClassifier
from naiveBayes import naivesBayesClassifier
import random

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

    knn = kNearestNeighborsClassifier(df)

    random.shuffle(df)
    border = int(len(df)*0.7)
    trainingData = df[0 : border]
    testData = df[border:]

    nb = naivesBayesClassifier()
    #nb.classifyData(trainingData, testData)

    crossValidation(df, nb)
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
    return (correct/len(testData))*100

def crossValidation(df, classifier):
    random.shuffle(df)

    # Create the 10 data packets
    dataParts = []
    for i in range(0, 10):
        if i == 0:
            dataParts.append(df[0:int(len(df) * 0.1)])
        else:
            dataParts.append(df[int(len(df) * (i/10)):int(len(df) * (i+1)/10)])
        print(len(dataParts[i]))

    # Test the classifier 10 times
    results = []
    for i in range(0, 10):
        testData = dataParts[i]
        trainingData = []

        for j in range(0, 10):
            if j != i:
                trainingData.extend(dataParts[j])
            else:
                continue
        results.append(reportStatistics(testData, classifier.classifyData(trainingData, testData)))

    # Average the results
    sum = 0
    for accuracy in results:
        sum += accuracy
    average = sum/len(results)

    print("The crossvalidated Accuracy "+classifier.name+" is: "+str(average))



if __name__ == "__main__":
    main()