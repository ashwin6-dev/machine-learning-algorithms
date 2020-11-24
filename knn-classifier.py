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


classifier = knn_classifier()
training_data_points = [
    [[255, 0, 0], "red"], 
    [[0, 255, 0], "green"], 
    [[0, 0, 255], "blue"],
    [[250, 5, 5], "red"],
    [[5, 250, 5], "green"],
    [[5, 5, 250], "blue"],
    [[245, 10, 10], "red"],
    [[10, 245, 10], "green"],
    [[10, 10, 245], "blue"],
]

for point in training_data_points:
    classifier.add_example(point[0], point[1])

print (classifier.classify([250, 0, 0], k = 3))
print (classifier.classify([100, 180, 50], k = 3))
print (classifier.classify([50, 50, 190], k = 3))