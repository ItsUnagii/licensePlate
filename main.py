import os
import segmentation
import joblib

curr_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(curr_dir, 'models/svc/svc.pkl')
model = joblib.load(model_dir)

classification_results = []
for char in segmentation.chars:
    # convert 1D array
    char = char.reshape(1, -1)
    result = model.predict(char)
    classification_results.append(result)

print(classification_results)

plate_string = ''
for predict in classification_results:
    plate_string += str(predict[0])

print(plate_string)

column_list_copy = segmentation.column_list[:]
segmentation.column_list.sort()
rightplate_string = ''
for each in segmentation.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print(rightplate_string)