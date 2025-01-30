import cv2
import dlib
import numpy as np
import os
import pickle

# Get the absolute path of the model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPE_PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
FACE_RECOGNITION_MODEL_PATH = os.path.join(BASE_DIR, "dlib_face_recognition_resnet_model_v1.dat")

def extract_features(image_path):
    # Load models
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_recognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Extract features
    for face in faces:
        shape = shape_predictor(gray, face)
        face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
        # print(np.array(face_descriptor))
        print("Facial Features extracted..")
        return np.array(face_descriptor)  # Return the 128D feature vector

    return None  # No face found

