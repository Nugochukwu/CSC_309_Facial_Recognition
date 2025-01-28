import os
from src.face_detection import detect_face_and_display
from src.feature_extraction import extract_features
from src.face_matching import match_faces

def main():
    pictures_dir = 'pictures/'  # Path to the pictures folder

    while True:
        print("\nWhere would you like to save the detected faces?")
        print("1. Detected Faces Directory (default: 'detected_faces/')")
        print("2. Dataset Directory (default: 'dataset/')")
        print("3. Exit")

        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '3':
            print("Exiting the program.")
            break

        image_output_dir = 'dataset/' if choice == '2' else 'detected_faces/'

        # Get list of images in the pictures folder
        image_files = [f for f in os.listdir(pictures_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print("No images found in the 'pictures/' directory.")
            continue

        # Process each image
        for image_name in image_files:
            image_path = os.path.join(pictures_dir, image_name)
            print(f"\nProcessing {image_name}...")

            try:
                detect_face_and_display(image_path, image_output_dir)
                extracted_features = extract_features(image_path)  # Pass the image path
                if extracted_features is not None:
                    match_faces(extracted_features)  # Pass extracted features
            except FileNotFoundError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
