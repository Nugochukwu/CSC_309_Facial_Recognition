import os
import pickle
from src.face_detection import detect_face_and_display
from src.feature_extraction import extract_features
from src.face_matching import match_faces


def main():
    authorised_pic_dir = 'authorised_personnel_data/'   # Path to the authorised_personnel folder
    pictures_dir = 'pictures/'                          # Path to the pictures folder
    dataset_dir = 'dataset/'                            # Path to the dataset folder
    detected_faces_dir = 'detected_faces/'              # Path to the detected_faces folder
    pickle_path = 'face_database.pkl'                   # Path to the face_database folder

    # Get list of images in the respective directories
    image_files = [f for f in os.listdir(pictures_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    detected_faces_files = [f for f in os.listdir(detected_faces_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Load existing known faces from pickle file if it exists
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            known_faces = pickle.load(file)
        print("✅ Known faces loaded successfully!")
    else:
        known_faces = {}  # Initialize as empty if file doesn't exist

    while True:
        print("\nWELCOME TO OUR FACIAL RECOGNITION PROJECT.\nWhat would you like to do?")
        print("1. Save detected faces to 'detected_faces/'")
        print("2. Save detected faces to 'dataset/'")
        print("3. clear database")
        print("4. save dataset to database")
        print("5. Match detected faces with dataset faces")
        print("0. Exit")

        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

        if choice == '0':
            print("Exiting the program.")
            break
        if choice == '3':
            print("Clearing database.")
            known_faces = {}
        # Determine output directory
        image_output_dir = detected_faces_dir if choice == '1' else dataset_dir if choice == '2' else None

        if not image_files:
            print("No images found in the 'pictures/' directory.")
            continue

        if choice in ('1', '2'):
            # Process each image within the pictures directory
            for image_name in image_files:
                image_path = os.path.join(pictures_dir, image_name)
                print(f"\nProcessing {image_name}...")

                try:
                    detect_face_and_display(image_path, image_output_dir, choice)
                    # Extracting facial features
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
        elif choice == '4':
            print(f"📌 Total known faces: {known_faces}")
            for image_name in dataset_files:
                image_path = os.path.join(dataset_dir, image_name)
                print(f"\nProcessing {image_name} from dataset...")
                try:
                    extracted_features = extract_features(image_path)
                    if extracted_features is not None:
                        if image_output_dir is not detected_faces_dir:
                            known_faces[image_name] = extracted_features
                            with open(pickle_path, "wb") as file:
                                pickle.dump(known_faces, file)
                            print(f"✅ Features for {image_name} added to the database.")
                        else:
                            continue
                    else:
                        print(f"❌ No face detected for {image_name}. Skipping.")

                except FileNotFoundError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
        elif choice == '5':
            if not known_faces:
                print("No known faces available. Add faces first!")
                continue

            for image_name in detected_faces_files:
                faces_path = os.path.join(detected_faces_dir, image_name)
                print(f"\nProcessing {image_name}...")
                match_faces(faces_path, known_faces)

if __name__ == "__main__":
    main()
