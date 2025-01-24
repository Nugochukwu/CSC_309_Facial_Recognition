import cv2
import matplotlib.pyplot as plt


def detect_face_and_display(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Check if the image was successfully loaded
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
    else:
        # Print the shape of the image
        print(f"Image loaded successfully. Dimensions: {img.shape}")
    print("Facial recognition Test")
    # Convert to grey scale # Has three output files
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dimensions of grey scale
    gray_image.shape # Has two output files
    # Load the pre-trained Haar Cascade classifier  (Designed specifically for detecting frontal faces in visual input.
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Perform Face Detection using the greyscale image
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    # Draw a Bounding Box
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    # convert image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # use matplotlib to display the image
    plt.figure(figsize=(20,10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()