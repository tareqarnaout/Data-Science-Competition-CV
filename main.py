import os
import cv2
import numpy as np
import face_recognition
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
from insightface.app import FaceAnalysis  # Import ArcFace
import json  # Import json module

# Initialize ArcFace model
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)

def preprocess_image(image, target_size=(224, 224), apply_hist_eq=True):
    """
    Preprocess an image for face recognition.

    Parameters:
    - image (numpy array): The input image (BGR format from OpenCV)
    - target_size (tuple): The desired output image size (width, height)
    - apply_hist_eq (bool): Whether to apply histogram equalization for contrast enhancement

    Returns:
    - preprocessed_image (numpy array): The preprocessed image ready for recognition
    """
    # Resize image to target size
    image_resized = cv2.resize(image, target_size)
    
    # Convert image from BGR to RGB (ArcFace uses RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Optionally apply histogram equalization (improves contrast)
    if apply_hist_eq:
        # Convert to grayscale first for histogram equalization
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        
        # Merge the equalized image back to RGB (create a 3-channel image)
        image_rgb[:, :, 0] = equalized_image  # Red channel
        image_rgb[:, :, 1] = equalized_image  # Green channel
        image_rgb[:, :, 2] = equalized_image  # Blue channel
    
    # Normalize pixel values to the range [0, 1]
    image_normalized = image_rgb / 255.0

    return image_normalized

def augment_encoding(encoding, num_augmentations=25):
    """
    Augment a face encoding by adding noise, scaling, and translating.

    Parameters:
    - encoding (numpy array): The original face encoding
    - num_augmentations (int): Number of augmentations to create

    Returns:
    - augmented_encodings (list): List of augmented encodings
    """
    augmented_encodings = []

    for _ in range(num_augmentations):
        # Add small random noise
        noise = np.random.normal(0, 0.01, encoding.shape)
        augmented_encoding = encoding + noise

        # Normalize to unit length as face_recognition expects
        augmented_encoding = augmented_encoding / np.linalg.norm(augmented_encoding)

        # Apply random scaling
        scale = np.random.uniform(0.9, 1.1)  # Scale between 0.9 and 1.1
        augmented_encoding = augmented_encoding * scale

        # Apply random translation
        translation = np.random.uniform(-0.01, 0.01, encoding.shape)
        augmented_encoding = augmented_encoding + translation

        # Normalize again to unit length
        augmented_encoding = augmented_encoding / np.linalg.norm(augmented_encoding)

        augmented_encodings.append(augmented_encoding)

    return augmented_encodings

def create_face_recognition_system():
    """
    Build a face recognition system from a 'faces_data' folder in the current directory
    
    Returns:
    model: Trained classifier
    encodings_list: List of face encodings
    names_list: List of corresponding names
    """
    # Use the faces_data folder in current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "faces_data")
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"Creating 'faces_data' folder in {current_dir}")
        os.makedirs(data_folder)
        print("Please add person folders with face images to the 'faces_data' folder and run again")
        return None, None, None
    
    # Lists to store data
    encodings_list = []
    names_list = []
    
    # Walk through the data folder
    for person_name in os.listdir(data_folder):
        person_dir = os.path.join(data_folder, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Processing images for {person_name}...")
        
        # Process each image in the person's directory
        for image_name in os.listdir(person_dir):
            # Skip non-image files
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(person_dir, image_name)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                
                # Preprocess image (optional)
                #image = preprocess_image(image)

                # Use ArcFace to get face embeddings
                faces = app.get(image)
                
                # Skip if no face found
                if len(faces) == 0:
                    print(f"  No face found in {image_name}, skipping.")
                    continue
                
                # Get face encodings (using only the first face if multiple are found)
                face_encoding = faces[0].embedding
                
                # Add to our lists
                encodings_list.append(face_encoding)
                names_list.append(person_name)
                
                print(f"  Processed {image_name}")
                
            except Exception as e:
                print(f"  Error processing {image_name}: {str(e)}")
    
    # Check if we have enough data
    if len(encodings_list) < 2:
        print("Not enough face data found. Need at least 2 images.")
        print("Please add more images to the 'faces_data' folder.")
        return None, None, None
    
    # Convert lists to numpy arrays
    X = np.array(encodings_list)
    y = np.array(names_list)
    
    # Data augmentation for limited data
    # Create slightly modified versions of existing face encodings
    if len(X) < 50:  # If we have very few samples
        print("Performing data augmentation...")
        augmented_X = []
        augmented_y = []
        
        for encoding, name in zip(X, y):
            # Add original
            augmented_X.append(encoding)
            augmented_y.append(name)
            
            # Add augmentations
            augmented_encodings = augment_encoding(encoding, num_augmentations=25)
            augmented_X.extend(augmented_encodings)
            augmented_y.extend([name] * len(augmented_encodings))
        
        X = np.array(augmented_X)
        y = np.array(augmented_y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a classifier
    model = KNeighborsClassifier(n_neighbors=1)  # KNN works well with limited data
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model, encodings_list, names_list

def process_test_images():
    """
    Find and process all images in the test_images folder in the current directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.join(current_dir, "test_images")
    
    # Check if folder exists
    if not os.path.exists(test_folder):
        print(f"Creating 'test_images' folder in {current_dir}")
        os.makedirs(test_folder)
        print("Please add images to recognize to the 'test_images' folder and run again")
        return
    
    # Create results folder if it doesn't exist
    results_folder = os.path.join(current_dir, "recognition_results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Train the model
    model, encodings, names = create_face_recognition_system()
    
    if model is None:
        return
    
    # Process each image in the test folder
    found_images = False
    results = {}
    for image_name in os.listdir(test_folder):
        # Skip non-image files
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        found_images = True
        image_path = os.path.join(test_folder, image_name)
        print(f"\nProcessing test image: {image_name}")
        
        # Recognize faces
        recognition_results = recognize_face_in_image(image_path, model)
        
        # Generate results
        if recognition_results:
            result_image = visualize_recognition(image_path, recognition_results)
            
            # Save result
            result_filename = f"result_{image_name}"
            result_path = os.path.join(results_folder, result_filename)
            cv2.imwrite(result_path, result_image)
            print(f"Results saved to {result_path}")
            
            # Print recognized people
            print("Recognized people:")
            for name, _ in recognition_results:
                print(f"- {name}")
                results[image_name] = name
        else:
            print("No faces found in this image.")
    
    if not found_images:
        print("No images found in test_images folder. Please add some images and run again.")
    
    # Save results to JSON file
    json_file_path = os.path.join(current_dir, "recognition_results.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"Results saved to {json_file_path}")

def recognize_face_in_image(image_path, model):
    """
    Recognize faces in a given image using the trained model
    
    Parameters:
    image_path (str): Path to the image
    model: Trained classifier
    
    Returns:
    List of (name, location) tuples for each face found
    """
    # Load image
    image = cv2.imread(image_path)
    
    # Use ArcFace to get face embeddings
    faces = app.get(image)
    
    if len(faces) == 0:
        return []
    
    results = []
    
    # Predict each face
    for face in faces:
        face_encoding = face.embedding
        face_location = face.bbox.astype(int)
        
        # Predict
        name = model.predict([face_encoding])[0]
        
        # Add to results
        results.append((name, face_location))
    
    return results

def visualize_recognition(image_path, recognition_results):
    """
    Draw boxes and names on the image for visualization
    
    Parameters:
    image_path (str): Path to the image
    recognition_results: List of (name, location) tuples
    
    Returns:
    The annotated image
    """
    # Load image with OpenCV (BGR format)
    image = cv2.imread(image_path)
    
    # Draw each face
    for name, (left, top, right, bottom) in recognition_results:
        # Draw a box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw a label
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    return image

if __name__ == "__main__":
    print("Face Recognition System")
    print("=======================")
    print("Looking for data in the current directory...")
    
    # Process all test images
    process_test_images()
    
    print("\nDone! Press any key to exit...")
    # Keep console window open on Windows
    time.sleep(1)