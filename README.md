# Face Detection Using Haar Cascade Classifier

This project implements a face detection system using Python's OpenCV library and the pre-trained Haar Cascade classifier. The program detects faces in an input image and displays the result with bounding boxes drawn around detected faces.

Further functionality to be added are face matching and feature extraction inorder to differentiate pictures not in the dataset and to calculate the accuracy of the prediction system


---

## Features

- Detect faces in an image using the Haar Cascade classifier.
- Display the detected faces with bounding boxes using Matplotlib.
- Easily adaptable to process multiple images.
- Modularized code for better reusability.

Prerequisites

Before running the program, ensure you have the following installed:

`Python (version 3.7 or higher)`

Required Python libraries:

`opencv-python`

`matplotlib`

To install these libraries, run:

``` 
pip install opencv-python matplotlib
```

Directory Structure
```
CSC_309_Facial_Recognition/
├── main.py                    # Main entry point
├── src/                       # Source code folder
│   ├── __init__.py            # Marks the directory as a package
│   ├── face_detection.py      # Contains the detect_face_and_display function
├── pictures/                  # Folder for images
│   └── man1.jpg               # Example input image
```
