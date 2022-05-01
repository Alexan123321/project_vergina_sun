import time
import numpy as np
from eigenfaces_module import EigenfacesServer

class TestSuite():
    server: EigenfacesServer
    labels: np.array([])

    def __init__(self, input_server: EigenfacesServer):
        self.server = input_server
        self.labels = []

    def computation_time_training(self, training_images: np.array([])):
        start = time.time()
        self.server.Train(training_images)
        duration = (time.time() - start)
        print(f"Training time: {(duration)/60:0.0f} minutes and {(duration)%60:0.0f} seconds.")

    def computation_time_classification(self, test_images: np.array([]), training_image_labels: np.array([])):
        start = time.time()
        for i in range(0, len(test_images)):
            self.labels = np.append(self.labels, self.server.Classify(test_images, training_image_labels))
        duration = time.time() - start
        print(self.labels)
        print(f"Classification time: {(duration)/60:0.0f} minutes and {(duration)%60:0.0f} seconds.")

    def prediction_accuracy(self, expected_labels: np.array([])):
        correct = 0
        for i in range (0, len(expected_labels)): 
            if self.labels[i] == expected_labels[i]:
                correct += 1
        print(f"Correctly classified: {correct / len(expected_labels) * 100}%")

    def print_test_report():
        pass