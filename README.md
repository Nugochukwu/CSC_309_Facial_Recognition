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

Directory Structure
```
CSC_309_Facial_Recognition/
├── main.py                    # Main entry point
├── src/                       # Source code folder
│   ├── __init__.py            # Marks the directory as a package
│   ├── face_detection.py      # Contains the detect_face_and_display function
├── pictures/                  # Folder for images
│   └── man1.jpg               # Example input image
├── Docs                       # Documentation
```
## Run the program using

```
python main.py
```
