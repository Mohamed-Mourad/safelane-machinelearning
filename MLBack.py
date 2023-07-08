# Imports
import cv2
import time
import datetime
import numpy as np
import onnxruntime
import googlemaps
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import storage
import os

# Initialize Firebase Admin SDK
cred = credentials.Certificate("safelane-7aa3e-firebase-adminsdk-nlijc-9186f3572f.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'safelane-7aa3e.appspot.com'
})
db = firestore.client()

# Initialize Maps Admin SDK
gmaps = googlemaps.Client(key="AIzaSyBvlM7JgI-AnixvzePwn_zxEhyemsgLxbs")


# Create storage bucket instance
storage_bucket = storage.bucket()


# Function to obtain the latitude and longitude
def get_current_location():
    current_location = gmaps.geolocate()
    if current_location is not None:
        return current_location['location']['lat'], current_location['location']['lng']
    else:
        deflat, deflong = 0.0, 0.0
        return deflat, deflong


# Function to save the obstacle to the Firebase Cloud Firestore database
def save_obstacle_to_database(status, lat, long, image_path, obstacle):
    # Read the image file as bytes
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    if obstacle == "normal":
        return

    if obstacle == "wet":
        severity = 1
    elif obstacle == "crack":
        severity = 2
    elif obstacle == "muddy":
        severity = 3
    elif obstacle == "pothole":
        severity = 4
    elif obstacle == "snowy":
        severity = 5

    # Check if the obstacle already exists in the database
    obstacles_ref = db.collection("obstacles")
    query = obstacles_ref.where("type", "==", obstacle).where("latitude", "==", lat).where("longitude", "==", long).limit(1)
    existing_obstacles = query.stream()

    if len(list(existing_obstacles)) > 0:
        print("Obstacle already exists in the database.")
        print(existing_obstacles)
        return

    # Upload the image to Firebase Storage
    image_blob = storage_bucket.blob(os.path.basename(image_path))
    with open(os.path.abspath(image_path), "rb") as image_file:
        image_blob.upload_from_file(image_file, content_type="image/jpeg")

    # Ensure the image upload is complete
    image_blob.reload()

    # Set a very long expiration time for the signed URL (e.g., 1 year from upload time)
    expiration_time = datetime.datetime.now() + datetime.timedelta(days=365)

    # Get the uploaded image URL
    image_url = image_blob.generate_signed_url(expiration=expiration_time)

    # Create a new obstacle document with its properties
    obstacle_data = {
        "fixed": status,
        "latitude": lat,
        "longitude": long,
        "obstacleIMG": image_url,
        "severity": severity,
        "type": obstacle,
    }

    # Add the obstacle to the Firestore collection
    db.collection("obstacles").add(obstacle_data)


# Load the saved ONNX model
model_path = 'obstaclesModel.onnx'
session = onnxruntime.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Define a dictionary mapping the obstacle classes to their names
obstacles_names = ["pothole", "crack", "muddy", "snowy", "wet", "normal"]

# Initialize the webcam and run the model on the live feed
cap = cv2.VideoCapture(0)  # Use the default camera (index 0)
i = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the input image
    resized_frame = cv2.resize(frame, (224, 224))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    input_data = np.expand_dims(gray_frame, axis=0)
    input_data = np.expand_dims(input_data, axis=3)
    input_data = input_data.astype(np.float32)

    # Run the inference
    output = session.run([output_name], {input_name: input_data})
    predicted_class_idx = np.argmax(output)

    # Get the predicted obstacle class name
    predicted_class_name = obstacles_names[predicted_class_idx]

    # Don't save "snowy" obstacles
    if predicted_class_name == "snowy":
        continue

    # Display the predicted obstacle class on the frame
    cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Obstacle Classification', frame)

    # Save the obstacle's image
    obstacle_image_path = "obstacle"+str(i)+".jpg"
    cv2.imwrite(obstacle_image_path, frame)

    # Call the function get_current_location() to get the current latitude and longitude
    latitude, longitude = get_current_location()

    # Call the function save_obstacle_to_database() to save the obstacle to the database
    obstacle_type = obstacles_names[predicted_class_idx]
    save_obstacle_to_database(False, latitude, longitude, obstacle_image_path, obstacle_type)

    # increment i for next file and sleep for 1 second
    i = i + 1
    time.sleep(1)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
