import cv2
import os
import matplotlib.pyplot as plt

# Global counter for unique face filenames
face_counter = 1

def detect_face_and_display(image_path, output_dir,choice, image_name):
    global face_counter  # Use a global counter to ensure sequential naming

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        print(f"No faces detected in {image_path}.")
        return

    print(f"Detected {len(faces)} face(s) in {image_path}")

    for (x, y, w, h) in faces:
        global face_counter
        face_img = img[y:y + h, x:x + w]
        choice = int(choice)
        if choice == 2:
            face_file_path = os.path.join(output_dir, f"{image_name}.png")
        elif choice == 1:
            face_file_path = os.path.join(output_dir, f"detected_face_{face_counter}.png")
        cv2.imwrite(face_file_path, face_img)
        print(f"Saved: {face_file_path}")
        face_counter += 1  # Increment the counter for the next face

    # Draw bounding boxes and display the image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_rgb)
    plt.axis('off')
   # plt.show()
    plt.close()
