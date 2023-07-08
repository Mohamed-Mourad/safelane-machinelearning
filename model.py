# Imports
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

#%%
# dataset path
dir_path = r"D:/Mado/University/Graduation Project/ML Model/road_obstacles"
size = (224, 224)
obstacles_names = ["pothole", "crack", "muddy", "snowy", "wet", "normal"]

# Initialize empty lists to store the images and their labels
images, labels = [], []

# Loop through the person names and read the images from their respective folders
for label, obstacle_name in enumerate(obstacles_names):
    folder_path = os.path.join(dir_path, obstacle_name)
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        # Read the image and resize it
        image = cv2.imread(image_path)
        image = cv2.resize(image, size)
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Normalize the pixel values of the image
        image = image / 255.0
        # Append the image and its label to the respective lists
        images.append(image)
        labels.append(label)

# Convert the image and label lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)


#%%
# Initialize empty lists to store the train and test images and labels
train_images = []
train_labels = []
test_images = []
test_labels = []

# Split the images and labels for each person separately
# to ensure that they all have the same number of train and test samples
for label in np.unique(labels):
    # Get the indices of images for the current label
    indices = np.where(labels == label)[0]
    # Split the indices into train and test sets
    train_indices, test_indices = train_test_split(indices, train_size=0.85, test_size=0.15, random_state=42)
    # Append the images and labels for the current label to the train and test sets
    train_images.extend(images[train_indices])
    train_labels.extend(labels[train_indices])
    test_images.extend(images[test_indices])
    test_labels.extend(labels[test_indices])

# Convert the train and test image and label lists to NumPy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Shuffle the data
train_data = np.column_stack((train_images.reshape(train_images.shape[0], -1), train_labels))
np.random.shuffle(train_data)
test_data = np.column_stack((test_images.reshape(test_images.shape[0], -1), test_labels))
np.random.shuffle(test_data)

# Split the train and test data into X_train, y_train and X_test, y_test
X_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)
X_test, y_test = test_data[:, :-1], test_data[:, -1].astype(int)


#%%
# Reshape X_train and X_test to have four dimensions for the Conv2D layer
X_train_reshaped = X_train.reshape(X_train.shape[0], 224, 224, 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], 224, 224, 1)

# Define the model architecture
obstaclesModel = Sequential([

    # Convolutional layers
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train_reshaped.shape[1:]),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flatten layer
    Flatten(),

    # Fully connected layers
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax'),

])

# Compile the model
obstaclesModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
obstaclesModel.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test))


#%%
# TESTING THE MODEL
test_loss, test_accuracy = obstaclesModel.evaluate(X_test_reshaped, y_test)
print('Model accuracy:', test_accuracy)


#%%
# VISUALIZING THE OUTPUT
# Make predictions on the test set using both models
y_predict = obstaclesModel.predict(X_test_reshaped)

# Randomly select 10 examples from the test set and get their predicted label for both models
sample_indices = random.sample(range(len(y_test)), 10)
sample_images = X_test_reshaped[sample_indices]
sample_labels = y_test[sample_indices]
sample_predictions = y_predict[sample_indices]

# Display the images and their predicted class labels for both models
for i in range(len(sample_indices)):
    image = sample_images[i]
    true_label = sample_labels[i]
    predicted_labelA = np.argmax(sample_predictions[i])
    print(f"True label: {obstacles_names[true_label]}, Predicted label A: {obstacles_names[predicted_labelA]}")
    plt.imshow(image, cmap='gray')
    plt.show()

#%%
# Exporting the model
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType

# Create example input
input_shape = (1, 224, 224, 1)  # Replace with the appropriate input shape
example_input = np.random.randn(*input_shape).astype(np.float32)

# Create initial type for the input
input_type = [('input', FloatTensorType(input_shape))]

# Convert the model to ONNX format
onnx_model = onnxmltools.convert.convert_keras(obstaclesModel, initial_types=input_type)

# Save the ONNX model
onnxmltools.utils.save_model(onnx_model, 'obstaclesModel.onnx')

#%%
