#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
from os.path import realpath as realpath

# Define paths to pre-trained models
face_model01 = realpath("../../.assets/models/Age_Gender_Detection_GFG/opencv_face_detector_uint8.pbtxt")
face_model02 = realpath("../../.assets/models/Age_Gender_Detection_GFG/opencv_face_detector_uint8.pb")
age_model01 = realpath("../../.assets/models/Age_Gender_Detection_GFG/age_deploy.prototxt")
age_model02 = realpath("../../.assets/models/Age_Gender_Detection_GFG/age_net.caffemodel")
gender_model01 = realpath("../../.assets/models/Age_Gender_Detection_GFG/gender_deploy.prototxt")
gender_model02 = realpath("../../.assets/models/Age_Gender_Detection_GFG/gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
IMG_WIDTH = 720
IMG_HEIGHT = 640
AGE_CAT = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_CAT = ["Male", "Female"]
CONFIDENCE_THRESHOLD = 0.7
BLOB_SIZE_01 = (300, 300)
BLOB_SIZE_02 = (227, 227)
BOUNDING_BOX_RGB_VAL = (0, 255, 0)
FIGSIZE = (7, 7)

# Function to load image
def load_image(image_path):
    real_path_to_image = realpath(image_path)
    image = cv2.imread(real_path_to_image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return image

def load_detection_models():
    face_net = cv2.dnn.readNet(face_model02, face_model01)
    age_net = cv2.dnn.readNet(age_model02, age_model01)
    gender_net = cv2.dnn.readNet(gender_model02, gender_model01)
    return face_net, age_net, gender_net

# Function to detect face and highlight with bounding box
def face_detection(mod_loaded_image, face_model=None):
    fr_cv = mod_loaded_image.copy()
    fr_height = fr_cv.shape[0]
    fr_width = fr_cv.shape[1]
    blob = cv2.dnn.blobFromImage(fr_cv, 1.0, BLOB_SIZE_01, [104, 117, 123], swapRB=True, crop=False)
    face_model.setInput(blob)
    detections = face_model.forward()
    # face bounding box creation
    face_boxes = []
    for i in range(detections.shape[2]):
        # bounding box creation if confidence > threshold
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            x1 = int(detections[0, 0, i, 3] * fr_width)
            y1 = int(detections[0, 0, i, 4] * fr_height)
            x2 = int(detections[0, 0, i, 5] * fr_width)
            y2 = int(detections[0, 0, i, 6] * fr_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(fr_cv, (x1, y1), (x2, y2), BOUNDING_BOX_RGB_VAL, int(round(fr_height / 150)), 8)
    return face_boxes, fr_cv


def gender_n_age_detection(face_boxes_arr, fr_cv, gender_model=None, age_model=None):
    # check if face was detected
    if not face_boxes_arr:
        print("No face detected")
    # loop for all faces detected
    for face_box in face_boxes_arr:
        # extract face as per the face_box
        face = fr_cv[max(0, face_box[1]-15):min(face_box[3]+15, fr_cv.shape[0]-1), max(0, face_box[0]-15):min(face_box[2]+15, fr_cv.shape[1]-1)]
        # extract the main blob
        blob = cv2.dnn.blobFromImage(face, 1.0, BLOB_SIZE_02, MODEL_MEAN_VALUES, swapRB=False)
        # prediction of gender
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = GENDER_CAT[gender_preds[0].argmax()]
        # prediction of age
        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = AGE_CAT[age_preds[0].argmax()]
        # put text of age and gender at top of box
        cv2.putText(
            fr_cv,
            f"{gender}, {age}",
            (face_box[0]-150, face_box[1]+10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (217, 0, 0),
            4,
            cv2.LINE_AA
        )
        plt.figure(figsize=FIGSIZE)
        plt.imshow((fr_cv))


def main(image_path=None):
    image_path = realpath(image_path)
    img_to_use = load_image(image_path)
    face_net, age_net, gender_net = load_detection_models()
    face_boxes, fr_cv = face_detection(mod_loaded_image=img_to_use, face_model=face_net)
    gender_n_age_detection(face_boxes_arr=face_boxes, fr_cv=fr_cv, gender_model=gender_net, age_model=age_net)

main("../../.assets/images/img-3-shopping-cart-js-starter-files.jpg")

if __name__ == "__main__":
    main()
