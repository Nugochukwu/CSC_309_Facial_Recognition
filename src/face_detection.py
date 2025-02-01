"""
    Detects a face puts a border around it crops it and
    exports the cropped image to an out output_dir.

    Parameters:
    - image_path: Path to the new image.
    - output_dir: path to the output directory.
    - choice: output_dir choice dictates th name used to save the exported file.

    Returns:
    - nil
"""


import cv2
import os
import matplotlib.pyplot as plt

# counter for unique face filenames
face_counter = 1


def detect_face_and_display(image_path, output_dir, image_name, choice):
    global face_counter

    # if selected output_dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # use cv2 to read the image stored in the given directory path
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    # convert the image to greyscale.
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar Cascade classifier for face detection.
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # Determines the number of faces detected in a picture
    if len(faces) == 0:
        print(f"No faces detected in {image_path}.")
        return

    # if no faces are detected in the picture
    print(f"Detected {len(faces)} face(s) in {image_path}")

    # Loop through all detected faces in the image.
    # where each detected face is represented as a rectangle with coordinates (x, y) and dimensions (w, h).
    for (x, y, w, h) in faces:
        global face_counter
        face_img = img[y:y + h, x:x + w]
        choice = int(choice)
        # deciding the directory to export the cropped picture to
        if choice == 2:
            face_file_path = os.path.join(output_dir, f"{image_name}.png")
            cv2.imwrite(face_file_path, face_img)
            print(f"Saved: {face_file_path}")
            # Increment the counter for the next face
            face_counter += 1
        elif choice == 1:
            face_file_path = os.path.join(output_dir, f"detected_face_{face_counter}.png")
            cv2.imwrite(face_file_path, face_img)
            print(f"Saved: {face_file_path}")
            face_counter += 1  # Increment the counter for the next face

    # Draw bounding boxes and display the image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert the image from BGR to RGB.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Create a new figure with a specified size in inches.
    plt.figure(figsize=(10, 5))
    # Display the image using Matplotlib.
    plt.imshow(img_rgb)
    # Remove axis labels and ticks for a cleaner display.
    plt.axis('off')
    # Show the image in a window.
    # plt.show()
    # Close the figure to free up memory.
    plt.close()
