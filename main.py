# python libraries used
# cv2: is the Python module for OpenCV, a popular computer vision library.
"""
# pickle(built-in): pickle is a built-in Python library used for serializing (saving) and deserializing (loading)
Python objects. It allows you to store Python data structures, such as dictionaries and lists, in a binary file and reload
them later
"""
# os: Allows fo interaction with the file system
# numpy: Handles numerical data efficiently. This is used for recording image arrays and face embeddings
# matplotlib.pyplot: Used to visualize images
# dlib: provides deep larning facial recognition

import os
import pickle
from src.face_detection import detect_face_and_display
from src.feature_extraction import extract_features
from src.face_matching import match_faces
# Check current directory and files
print("📂 Current Directory:", os.getcwd())
print("📜 Files:", os.listdir(os.getcwd()))


def main():
    # directory definitions
    authorised_pic_dir = 'authorised_personnel_data/'   # Path to the authorised_personnel folder
    pictures_dir = 'pictures_for_detection/'                          # Path to the pictures folder
    dataset_dir = 'dataset/'                            # Path to the dataset folder
    detected_faces_dir = 'detected_faces/'              # Path to the detected_faces folder
    pickle_path = 'face_database.pkl'                   # Path to the face_database folder

    # Get list of images in the respective directories
    # I might need to rework this section
    image_files = [f for f in os.listdir(pictures_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files += [f for f in os.listdir(authorised_pic_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    detected_faces_files = [f for f in os.listdir(detected_faces_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Load existing known faces from pickle file if it exists
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            known_faces = pickle.load(file)
        print("✅ Known faces loaded successfully!")
    else:
        # Initialize as empty if file doesn't exist
        known_faces = {}
    # Menu Options
    while True:
        print("\nWELCOME TO OUR FACIAL RECOGNITION SYSTEM MENU5.\nWhat would you like to do?")
        print("1. Save new faces to 'detected_faces/'")
        print("2. Save authorised faces to 'dataset/'")
        print("3. Clear database")
        print("4. Save dataset to database")
        print("5. Match detected faces with dataset faces")
        print("0. Exit")
        print("Restart Program if you encounter any problems")

        choice = input("Enter your choice (1, 2, 3, 4, or 5): ").strip()
        # stop the program
        if choice == '0':
            print("Exiting the program.")
            break
        # Clear the database
        elif choice == '3':
            print("Clearing database.")
            known_faces = {}
        # Match faces in detected_faces_dir with dataset_dir
        elif choice == '5':
            if not known_faces:
                print("No known faces available. Add faces first!")
                continue
            for image_name in detected_faces_files:
                faces_path = os.path.join(detected_faces_dir, image_name)
                print(f"\nProcessing {image_name}...")
                match_faces(faces_path)
        # Choose cropped detected pictures output directory
        elif choice == '1' or choice == '2':
            image_output_dir = detected_faces_dir if choice == '1' else dataset_dir
            pic_folder = pictures_dir if choice == '1' else authorised_pic_dir

            # Check if the directories are valid
            if not image_files:
                print(f"No images found in the '{pic_folder}' directory.")
                continue

            # Process images in the selected folder
            for image_name in image_files:
                image_path = os.path.join(pic_folder, image_name)
                print(f"\nProcessing {image_name}...")

                try:
                    detect_face_and_display(image_path, image_output_dir, image_name, choice)
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
        # Add pictures in the dataset_dir to the database
        elif choice == '4':
            for image_name in dataset_files:
                image_path = os.path.join(dataset_dir, image_name)
                print(f"\nProcessing {image_name} from dataset...")
                try:
                    extracted_features = extract_features(image_path)
                    if extracted_features is not None:
                        known_faces[image_name] = extracted_features
                        with open(pickle_path, "wb") as file:
                            pickle.dump(known_faces, file)
                        print(f"✅ Features for {image_name} added to the database.")
                    else:
                        print(f"❌ No face detected for {image_name}. Skipping.")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                print(f"📌 Total known faces: {len(known_faces)}")
        # If choice is invalid re-choose a menu option
        else:
            print("Invalid choice. Please select '1', '2', '3', '4', or '5'.")


if __name__ == "__main__":
    main()
