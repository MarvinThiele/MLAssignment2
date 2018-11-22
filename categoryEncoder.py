
class categoryEncoder:
    borders = {}

    def __init__(self):
        pass

    def fit(self, dataframe):
        self.borders = self.equalHeightBinning(dataframe)

    def transform(self, dataframe):
        transformedDataframe = []
        for instance in dataframe:
            transformedInstance = []
            for i, feature in enumerate(instance):
                if i == 57:
                    transformedInstance.append(feature)
                elif feature == 0:
                    transformedInstance.append(0)
                elif feature <= self.borders[i][0]:
                    transformedInstance.append(1)
                elif feature <= self.borders[i][1]:
                    transformedInstance.append(2)
                elif feature <= self.borders[i][2]:
                    transformedInstance.append(3)
                elif feature <= self.borders[i][3]:
                    transformedInstance.append(4)
                else:
                    transformedInstance.append(5)
            transformedDataframe.append(transformedInstance)
        return transformedDataframe

    def fitTransform(self, dataframe):
        self.fit(dataframe)
        return self.transform(dataframe)

    def equalHeightBinning(self, dataframe):
        borders = {}
        borderPercentages = [0.2, 0.4, 0.6, 0.8]
        for i in range(0, 57):
            values = []
            for instance in dataframe:
                if instance[i] != 0:
                    values.append(instance[i])
            values = sorted(values)
            bd = []
            for border in borderPercentages:
                limit = int(len(values) * border)
                bd.append(values[limit])
            borders[i] = bd
        return borders