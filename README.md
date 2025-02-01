# Face Detection Using Haar Cascade Classifier

This project demonstrates a face detection system implemented using Python's OpenCV library and a pre-trained Haar Cascade classifier. The program detects faces in an input image and displays the results by drawing bounding boxes around the detected faces.

Future enhancements include implementing face matching and feature extraction to distinguish images not present in the dataset, as well as evaluating the system's accuracy to ensure reliable predictions.


---

## Features

- Detect faces in an image using the Haar Cascade classifier.
- Display the detected faces with bounding boxes using Matplotlib.
- Easily adaptable to process multiple images.
- Modularized code for better reusability.

Prerequisites

Before running the program, ensure you have the following installed:

`Git Repository files`

`Python (version 3.7 or higher)`

Required Python libraries:

`opencv-python` ,
`matplotlib`

To install these libraries, run:

``` 
pip install opencv-python matplotlib
```
You also need 
`dlib`
To install this library, run
```
pip install dlib
# This method should work if you have all necessary dependencies installed, like CMake and a C++ compiler.
# If you're using Windows, dlib might require you to have Visual Studio or other build tools installed.
```
if you're encountering issues with building it from source, you can use `.whl (wheel)` files, which are precompiled binaries.
---
Go to an unofficial dlib binaries repository:
One popular repository is: https://github.com/jeremiah-williams/dlib-windows-binaries.
The .whl file corresponds to your Python version and operating system. For example, if you're using Python 3.8 on a 64-bit Windows machine, you should look for a file named like dlib-19.22.0-cp38-cp38-win_amd64.whl

Run `python --version`to check which version of Python you're using, and make sure the `.whl` file matches your Python version.
The `cp38` part refers to `Python 3.8`, and `cp39` would be for `Python 3.9`. If you're unsure about your Python version, you can find it by running:
```
python --version
```

Directory Structure
```
CSC_309_Facial_Recognition/

├── dataset                    # Faces that are allowed access
├── detected_faces             # Faces detected
├── pictures/                  # Folder for images
│   └── man1.jpg               # Example input image
├── Docs                       # Documentation
├── src/                       # Source code folder
│   ├── __init__.py            # Marks the directory as a package
│   ├── face_detection.py                           #
    ├── face_extraction.py                          #
    ├── face_matching.py                            # Contains the detect_face_and_display function
    ├── dlib_face_recognition_resnet_model_v1.dat   #
    ├── shape_predictor_68_face_landmarks.dat       #
    ├── dlib_face_recognition_resnet_model_v1.dat   #
    ├── dlib_face_recognition_resnet_model_v1.dat   #
├── main.py                    # Main entry point

```
## Run the program using

```
python main.py
```
