# python 3.7
# Scikit-learn ver. 0.23.2
from imblearn import pipeline
from scipy.sparse import data
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
# Imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
# OpenCv
import cv2
# matplotlib 3.3.1
from matplotlib import pyplot
# Numpy
import numpy as np
from sklearn.utils import multiclass
# DataLoader
from DataLoader import Dataset
# from DataLoader import MiniBatch
# Pickle
import pickle


# Hyperparameters
epochs = 2
batch_size = 16
max_each_class = 1000

def main():
    train()

def train():
    
    data_loader = Dataset('Dataset\\train\\', batch_size=batch_size, max_each_class=max_each_class)
    # Verify Sizes
    print(f'Data Loader Length: {len(data_loader)}')
    # Declare Model
    model = SGDClassifier(random_state=0, loss='log', penalty='l2') # verbose=1
    # Train Loop
    # Set Rounds Per Epoch
    rounds_per_epoch = int(len(data_loader)/batch_size)
    # Test Data
    test_x, test_y = data_loader.get_test_data()

    for epoch in range(epochs):
        for round in range(rounds_per_epoch):
            data, label = data_loader.get_next_batch()
            model.partial_fit(data, label, classes=data_loader.num_classes_list)
            if round % 15 == 0:
                print(f'Batches Checked: {round}/{rounds_per_epoch}')
        print(f'Epoch: {epoch}/{epochs}')
        test(model, test_x, test_y)
        data_loader.reset_index()
        # plot_confusion_matrix(model, test_x, test_y)
    plot_confusion_matrix(model, test_x, test_y)
    pyplot.show()
    
    # for i in range(len(test_y)):
    #     check_image(test_x[i], test_y[i], model)

    camera_test(model)

def main_2():

    batch_size = 7200


    data_loader = Dataset('Recaptcha_Data_2\\', batch_size=batch_size, max_each_class=max_each_class)
    # Verify Sizes
    print(f'Data Loader Length: {len(data_loader)}')
    # Declare Model
    model = LinearSVC(random_state=0, loss='hinge', penalty='l2', verbose=1, max_iter=10000) # verbose=1
    # Test Data
    test_x, test_y = data_loader.get_test_data()

    # Train
    train_x, train_y = data_loader.get_next_batch()
    print(f'Train Length: {len(train_x)}')
    model.fit(train_x, train_y)
    plot_confusion_matrix(model, test_x, test_y)
    pyplot.show()


    test(model, test_x, test_y)

    for i in range(len(test_y)):
        check_image(test_x[i], test_y[i], model)


def camera_test(model):
    label_dict = {
            'paper': 0,
            'rock': 1,
            'scissors': 2,
    }
    # Reverse label_dict
    label_dict = {v: k for k, v in label_dict.items()}

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (300, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Canny(frame, 70, 90)
        frame = frame.reshape(1, -1)
        predictions = model.predict(frame)
        print(label_dict[predictions[0]])
        frame = frame.reshape(300, 300, 1)
        cv2.imshow('Image', frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def test(model, test_x_final, test_y_final):
    preds = model.predict(test_x_final)
    correct = 0
    incorrect = 0
    for pred, gt in zip(preds, test_y_final):
        if pred == gt: correct += 1
        else: incorrect += 1
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")
    pyplot.show()





def check_image(image, label, model):
    # Labels
    # Labels
    label_dict = {
            'paper': 0,
            'rock': 1,
            'scissors': 2,
    }
    # Reverse label_dict
    reverse_label_dict = {v: k for k, v in label_dict.items()}


    print(f'Label: {reverse_label_dict[label]}')
    predictions = model.predict(image.reshape(1, -1))
    print(f'Prediction {reverse_label_dict[predictions[0]]}')
    image = image.reshape(300, 300, 1)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # image = cv2.imread('Dataset\\train\\rock\\rock01-000.png')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.Canny(image, 70, 90)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    main() # Train Working
    # main_2() # Test Working