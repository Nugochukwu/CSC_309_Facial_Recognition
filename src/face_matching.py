"""
    Compares a new face with known faces and returns the closest match.

    Parameters:
    - new_image_path: Path to the new image.
    - known_faces: Dictionary of stored face feature vectors {name: feature_vector}.
    - threshold: Maximum distance for a match (default: 0.6).

    Returns:
    - Best match name or "Unknown"
"""

import os
import pickle
import numpy as np

# importing extract_features module
from src.feature_extraction import extract_features

# initializing the path to pickle database
pickle_path = "face_database.pkl"

# Load known faces from pickle file/database
if os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        known_faces = pickle.load(f)
    print("✅ Known faces loaded successfully!")
else:
    print(f"❌ Error: '{pickle_path}' file not found! Creating a new database...")
    # Initialize an empty dictionary
    known_faces = {}


def match_faces(new_image_path, threshold=0.4):
    # Extract features from new image
    new_features = extract_features(new_image_path)
    # Check if the new features are None (no face detected)
    if new_features is None:
        print(f"❌ No face detected in {new_image_path}.")
        # return if no features were detected
        return "Unknown", None, None

    print(f"Features extracted for {new_image_path}: {type(new_features)}")

    # Initialize the best match as none
    best_match = None
    # Initialize the minimum distance as infinity
    min_distance = float("inf")

    if not known_faces:
        print("❌ No known faces to match against.")
        return "Unknown", None, None

    for name, stored_features in known_faces.items():
        # Ensure stored features are valid
        if stored_features is None or not isinstance(stored_features, np.ndarray):
            print(f"❌ Skipping {name} due to invalid stored features.")
            continue  # Skip invalid stored features

        # Compute distance & similarity
        # Euclidean distance
        distance = np.linalg.norm(new_features - stored_features)
        # Euclidean similarity
        similarity = 1 / (1 + distance)
        print(f"Matching {new_image_path} with {name}... (Distance: {distance:.4f}, Similarity: {similarity:.4f})")
        # Update the closest match
        if distance < min_distance:
            min_distance = distance
            best_match = name

    # Check if the best match is within the threshold
    if min_distance < threshold:
        print(f"✅ Best match to {new_image_path} is: {best_match} (Distance: {min_distance:.4f}, Similarity: {1 / (1 + min_distance):.4f})")
        return best_match, min_distance, 1 / (1 + min_distance)
    else:
        print(f"❌ No match found for {new_image_path} within the threshold.")
        return "Unknown", min_distance, 1 / (1 + min_distance)
