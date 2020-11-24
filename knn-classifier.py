import numpy as np


def get_mode(l):
    mode = ""
    max_count = 0
    count = {}

    for i in l:
        if i not in count:
            count[i] = 0
        count[i] += 1

        if count[i] > max_count:
            max_count = count[i]
            mode = i

    return mode

class knn_classifier:
    def __init__(self):
        self.data_points = []
        self.classifications = []

    def add_example(self, data_point, classification):
        #Adding training data points

        #self.data_points contain the data points themseleves, self.classification contain their respective classifications
        self.data_points.append(data_point)
        self.classifications.append(classification)

    def classify(self, input, k = 3):
        #Classifies new data
        classification = sorted(self.classifications, key = lambda x: np.linalg.norm(np.subtract(input, self.data_points[self.classifications.index(x)])))[:k]
        #The above line may seem confusing. It sorts self.classification by the euclidean distance between each classification's respective data point and the input data point
        #"classification" is ultimately sliced to contain the classifications of the k closest data points

        # Returning the final classification
        return get_mode(classification)
