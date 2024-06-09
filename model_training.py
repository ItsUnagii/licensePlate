import os
import numpy as np
from sklearn.svm import SVC # use support vector classifier for machine learning. might be better alternatives?
from sklearn.model_selection import cross_val_score
import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

letters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]


def read_training_data(training_directory):
    image_data = []
    target_data = []
    
    for letter in letters:
        for each in range(10):
            image_path = os.path.join(training_directory, letter, letter + '_' + str(each) + '.jpg') # path to training images
            img_details = imread(image_path, as_gray=True) # read image as grayscale
            binary_image = img_details < threshold_otsu(img_details) # convert image to binary
            flat_bin_image = binary_image.reshape(-1) # flatten image (machine learning needs 1D array?)
            image_data.append(flat_bin_image)
            target_data.append(letter)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label): # prevent overfitting (model trained too specific on training data)
    # cross validation divides data into subsets and uses one to validate the model trained on the others
    # num_of_fold = number of subsets
    # e.g. 5-fold cross validation: data divided into 5 subsets, model trained on 4 and validated on 1
    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold)
    print("Cross Validation Result for ", num_of_fold, "-fold")
    print(accuracy_result * 100)

curr_dir = os.path.dirname(os.path.realpath(__file__))
training_dataset_dir = os.path.join(curr_dir, 'training_data')
image_data, target_data = read_training_data(training_dataset_dir)

svc_model = SVC(kernel='linear', probability=True) # no clue why the kernel is linear. copilot autofilled for me

cross_validation(svc_model, 4, image_data, target_data) # 4-fold cross validation. arbitrary choice

svc_model.fit(image_data, target_data) # train it!!

save_directory = os.path.join(curr_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

joblib.dump(svc_model, save_directory + '/svc.pkl') # save model to file