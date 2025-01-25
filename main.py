from src.face_detection import detect_face_and_display
from src.feature_extraction import extract_features
from src.face_matching import match_faces

def main():
    # Define the image path
    image_path = 'pictures/man1.jpg'
    imageOutput_dir = 'dataset/'
    try:
        detect_face_and_display(image_path,imageOutput_dir)
        extract_features()
        match_faces()
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()