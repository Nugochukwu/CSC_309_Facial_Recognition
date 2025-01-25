from src.face_detection import detect_face_and_display
from src.feature_extraction import extract_features
from src.face_matching import match_faces

        # here we will add the functionality to detect a face,
        # extract its features
        # then match newly detected faces with the ones stored in our dataset.
# Main function
def main():
    while True:
        # Ask the user where to save the detected faces
        print("\nWhere would you like to save the detected faces?")
        print("1. Detected Faces Directory (default: 'detected_faces/')")
        print("2. Dataset Directory (default: 'dataset/')")
        print("3. Exit")

        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '3':  # Exit condition
            print("Exiting the program.")
            break

        if choice == '2':
            image_output_dir = 'dataset/'
        else:
            image_output_dir = 'detected_faces/'

        # Define the image path
        image_path = 'pictures/man1.jpg'

        try:
            # Detect faces, extract features, and match faces
            print(f"Saving detected faces to: {image_output_dir}")
            detect_face_and_display(image_path, image_output_dir)
            extract_features()
            match_faces()
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
