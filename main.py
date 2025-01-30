import os
import pickle
from src.face_detection import detect_face_and_display
from src.feature_extraction import extract_features
from src.face_matching import match_faces


def main():
    pictures_dir = 'pictures/'  # Path to the pictures folder
    dataset_dir = 'dataset/'
    detected_faces_dir = 'detected_faces/'
    pickle_path = 'face_database.pkl'

    # Load existing known faces from pickle file if it exists
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            known_faces = pickle.load(file)
        print("✅ Known faces loaded successfully!")
    else:
        known_faces = {}  # Initialize as empty if file doesn't exist

    while True:
        print("\nWhere would you like to save the detected faces?")
        print("1. Detected Faces Directory (default: 'detected_faces/')")
        print("2. Dataset Directory (default: 'dataset/')")
        print("3. Match Detected faces with faces stored in the dataset")
        print("4. Exit")

        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '4':
            print("Exiting the program.")
            break

        # Save to detected faces directory (default)
        if choice == '1':
            image_output_dir = detected_faces_dir
        elif choice == '2':
            image_output_dir = dataset_dir
        elif choice == '3':
            face_files = [f for f in os.listdir(detected_faces_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_name in face_files:
                faces_path = os.path.join(detected_faces_dir, image_name)
                print(f"\nProcessing {image_name}...")
                match_faces(faces_path, known_faces)
            continue

        # Get list of images in the pictures folder
        image_files = [f for f in os.listdir(pictures_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print("No images found in the 'pictures/' directory.")
            continue

        # Process each image within the pictures directory
        for image_name in image_files:
            image_path = os.path.join(pictures_dir, image_name)
            print(f"\nProcessing {image_name}...")

            try:
                detect_face_and_display(image_path, image_output_dir)
                # Extracting facial features
                extracted_features = extract_features(image_path)  # Pass the image path

                # Only save features if they were extracted successfully
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

if __name__ == "__main__":
    main()
