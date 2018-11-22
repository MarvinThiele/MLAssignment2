from nearestNeighbors import kNearestNeighborsClassifier

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

    for instance in df:
        score = knn.classifyInstance(instance)
        pass

if __name__ == "__main__":
    main()