"""import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                          shear_range = 0.2,
                                          zoom_range = 0.2,
                                          horizontal_flip = True)
training_set = train_datagen.flow_from_directory('train/train',
                                                 target_size=(64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
"""
""""
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import zipfile

# Step 1: Prepare the Dataset
dataset_path = "dogs-vs-cats.zip"  # Replace with your dataset path

# Unzipping the dataset
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall("data")

base_dir = "data/dogs-vs-cats"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test1")

# Step 2: Data Preprocessing
# Define ImageDataGenerators for augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Building the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust epochs as needed
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Step 7: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'")
"""
# Import necessary libraries
"""
import os
import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Step 1: Unzip and Organize the Dataset
def unzip_and_organize():
    # Unzipping the training dataset
    with zipfile.ZipFile('train.zip', 'r') as zip_ref:
        zip_ref.extractall('data/train')

    # Unzipping the test dataset
    with zipfile.ZipFile('test1.zip', 'r') as zip_ref:
        zip_ref.extractall('data/test1')

    # Create directories for organized training data
    os.makedirs('data/train/dogs', exist_ok=True)
    os.makedirs('data/train/cats', exist_ok=True)

    # Move training images into respective folders
    for filename in os.listdir('data/train'):
        if filename.startswith('dog'):
            shutil.move(f'data/train/{filename}', 'data/train/dogs/')
        elif filename.startswith('cat'):
            shutil.move(f'data/train/{filename}', 'data/train/cats/')

# Unzip and organize dataset
unzip_and_organize()

# Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust epochs as needed
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory='data/',
    classes=['test1'],  # Test directory
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

# Map predictions to filenames
filenames = test_generator.filenames
results = list(zip(filenames, predicted_classes))

# Save predictions to a CSV file
import pandas as pd
submission = pd.DataFrame(results, columns=['Filename', 'Prediction'])
submission['Prediction'] = submission['Prediction'].map({1: 'Dog', 0: 'Cat'})
submission.to_csv('dog_vs_cat_predictions.csv', index=False)
print("Predictions saved to 'dog_vs_cat_predictions.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
"""
# Import necessary libraries
"""import os
import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Unzip and Organize the Dataset
def unzip_and_organize():
    # Unzipping the training dataset
    with zipfile.ZipFile('train.zip', 'r') as zip_ref:
        zip_ref.extractall('data/train')

    # Unzipping the test dataset
    with zipfile.ZipFile('test1.zip', 'r') as zip_ref:
        zip_ref.extractall('data/test1')

    # Create directories for organized training data
    os.makedirs('data/train/dogs', exist_ok=True)
    os.makedirs('data/train/cats', exist_ok=True)

    # Move training images into respective folders
    for filename in os.listdir('data/train'):
        if filename.startswith('dog'):
            shutil.move(f'data/train/{filename}', 'data/train/dogs/')
        elif filename.startswith('cat'):
            shutil.move(f'data/train/{filename}', 'data/train/cats/')

# Unzip and organize dataset
unzip_and_organize()

# Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust epochs as needed
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory='data/',
    classes=['test1'],  # Test directory
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

# Map predictions to filenames
filenames = test_generator.filenames
file_ids = [int(f.split('/')[-1].split('.')[0]) for f in filenames]  # Extract file IDs from filenames

# Load sampleSubmission.csv to update predictions
sample_submission = pd.read_csv('sampleSubmission.csv')
sample_submission['label'] = [1 if pred > 0.5 else 0 for pred in predictions]  # Update with model predictions

# Save updated predictions to a CSV file
sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
"""
# Import necessary libraries
"""import os
import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Unzip and Organize the Dataset
def unzip_and_organize():
    # Unzipping the training dataset
    with zipfile.ZipFile('train.zip', 'r') as zip_ref:
        zip_ref.extractall('data/train')

    # Unzipping the test dataset
    with zipfile.ZipFile('test1.zip', 'r') as zip_ref:
        zip_ref.extractall('data/test1')

    # Create directories for organized training data
    os.makedirs('data/train/dogs', exist_ok=True)
    os.makedirs('data/train/cats', exist_ok=True)

    # Move training images into respective folders (dogs and cats)
    for filename in os.listdir('data/train'):
        if filename.startswith('dog'):
            shutil.move(f'data/train/{filename}', 'data/train/dogs/')
        elif filename.startswith('cat'):
            shutil.move(f'data/train/{filename}', 'data/train/cats/')

# Unzip and organize dataset
unzip_and_organize()

# Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust epochs as needed
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory='data/test1',  # Corrected path for test images
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

# Map predictions to filenames
filenames = test_generator.filenames
file_ids = [int(f.split('/')[-1].split('.')[0]) for f in filenames]  # Extract file IDs from filenames

# Load sampleSubmission.csv to update predictions
sample_submission = pd.read_csv('sampleSubmission.csv')
sample_submission['label'] = [1 if pred > 0.5 else 0 for pred in predictions]  # Update with model predictions

# Save updated predictions to a CSV file
sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
"""

"""import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=1,  # Adjust epochs as needed
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,  # Corrected path for test images
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

# Map predictions to filenames
filenames = test_generator.filenames
file_ids = [int(f.split('/')[-1].split('.')[0]) for f in filenames]  # Extract file IDs from filenames

# Load sampleSubmission.csv to update predictions
sample_submission = pd.read_csv('sampleSubmission.csv')
sample_submission['label'] = [1 if pred > 0.5 else 0 for pred in predictions]  # Update with model predictions

# Save updated predictions to a CSV file
sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
"""
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=1,  # Set epochs to 2 for faster training
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,  # Corrected path for test images
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

# Map predictions to filenames
filenames = test_generator.filenames
file_ids = [int(f.split('/')[-1].split('.')[0]) for f in filenames]  # Extract file IDs from filenames

# Load sampleSubmission.csv to update predictions
sample_submission = pd.read_csv('sampleSubmission.csv')
sample_submission['label'] = predicted_classes  # Update with model predictions

# Save updated predictions to a CSV file
sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
"""
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'  # Assuming the test images are directly in this folder

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=1,  # Set epochs to 2 for faster training
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images for single-folder test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Modified flow for test data (directly from test_dir)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False  # Important: do not shuffle for predictions
)

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

# Map predictions to filenames
filenames = test_generator.filenames
file_ids = [int(f.split('/')[-1].split('.')[0]) for f in filenames]  # Extract file IDs from filenames

# Load sampleSubmission.csv to update predictions
sample_submission = pd.read_csv('sampleSubmission.csv')
sample_submission['label'] = predicted_classes  # Update with model predictions

# Save updated predictions to a CSV file
sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
"""
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'  # Make sure test images are directly in this folder

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=1,  # Set epochs to 1 for quicker training
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,  # Corrected path for test images
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

# Check if test images are correctly found
print(f"Found {test_generator.samples} images for prediction.")

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

# Map predictions to filenames
filenames = test_generator.filenames
file_ids = [int(f.split('/')[-1].split('.')[0]) for f in filenames]  # Extract file IDs from filenames

# Load sampleSubmission.csv to update predictions
sample_submission = pd.read_csv('sampleSubmission.csv')
sample_submission['label'] = predicted_classes  # Update with model predictions

# Save updated predictions to a CSV file
sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'  # Ensure test images are directly in this folder

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=1,  # Set epochs to 1 for quicker training
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,  # Corrected path for test images
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

# Check if test images are correctly found
print(f"Found {test_generator.samples} images for prediction.")

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

# Map predictions to filenames
filenames = test_generator.filenames
file_ids = [int(f.split('.')[0]) for f in filenames]  # Extract file IDs from filenames (assuming they are like 1.jpg, 2.jpg, etc.)

# Load sampleSubmission.csv to update predictions
sample_submission = pd.read_csv('sampleSubmission.csv')
sample_submission['label'] = predicted_classes  # Update with model predictions

# Save updated predictions to a CSV file
sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")"""
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'  # Ensure test images are directly in this folder

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=1,  # Set epochs to 1 for quicker training
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,  # Corrected path for test images
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

# Check if test images are correctly found
print(f"Found {test_generator.samples} images for prediction.")

# Ensure that test images are correctly loaded
if test_generator.samples == 0:
    print("No test images found. Please check the directory structure of your test data.")
else:
    # Generate predictions
    predictions = model.predict(test_generator)
    predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

    # Map predictions to filenames
    filenames = test_generator.filenames
    file_ids = [int(f.split('.')[0]) for f in filenames]  # Extract file IDs from filenames (assuming they are like 1.jpg, 2.jpg, etc.)

    # Load sampleSubmission.csv to update predictions
    sample_submission = pd.read_csv('sampleSubmission.csv')
    sample_submission['label'] = predicted_classes  # Update with model predictions

    # Save updated predictions to a CSV file
    sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
    print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")

"""
'''
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'  # Ensure test images are directly in this folder

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=1,  # Set epochs to 1 for quicker training
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict
# Preprocess test images
test_images = []
test_filenames = os.listdir(test_dir)  # List all files in the test directory

# Make sure test files exist
if len(test_filenames) == 0:
    print("No files found in the test directory.")
else:
    # Load and preprocess each test image
    for filename in test_filenames:
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(test_dir, filename)
            img = load_img(img_path, target_size=(150, 150))  # Load image and resize
            img_array = img_to_array(img) / 255.0  # Convert to array and normalize
            test_images.append(img_array)
    
    # Convert list of images into a NumPy array
    test_images = np.array(test_images)

    # Predict classes
    predictions = model.predict(test_images)
    predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

    # Map predictions to filenames
    file_ids = [int(f.split('.')[0]) for f in test_filenames]  # Extract file IDs from filenames (assuming they are like 1.jpg, 2.jpg, etc.)

    # Load sampleSubmission.csv to update predictions
    sample_submission = pd.read_csv('sampleSubmission.csv')
    sample_submission['label'] = predicted_classes  # Update with model predictions

    # Save updated predictions to a CSV file
    sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
    print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
'''
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'  # Ensure test images are directly in this folder

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=1,  # Set epochs to 1 for quicker training
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict using ImageDataGenerator (avoid memory issues)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,  # Corrected path for test images
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

# Check if test images are correctly found
print(f"Found {test_generator.samples} images for prediction.")

# Generate predictions in batches
predictions = model.predict(test_generator)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

# Map predictions to filenames
filenames = test_generator.filenames
file_ids = [int(f.split('.')[0]) for f in filenames]  # Extract file IDs from filenames (assuming they are like 1.jpg, 2.jpg, etc.)

# Load sampleSubmission.csv to update predictions
sample_submission = pd.read_csv('sampleSubmission.csv')
sample_submission['label'] = predicted_classes  # Update with model predictions

# Save updated predictions to a CSV file
sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
'''
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pandas as pd

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'  # Ensure test images are directly in this folder

# Step 2: Data Preprocessing
# Image augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)

# Generate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    epochs=1,  # Set epochs to 1 for quicker training
    validation_data=validation_generator
)

# Step 6: Evaluate the Model
print("Evaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 7: Test and Predict using ImageDataGenerator (avoid memory issues)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Check if the test folder is correctly structured
test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]  # Get all .jpg files in test folder
print(f"Found {len(test_images)} test images.")

# Check if there are images in the test folder
if len(test_images) > 0:
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,  # Corrected path for test images
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,  # No labels for test data
        shuffle=False
    )

    # Generate predictions
    predictions = model.predict(test_generator)
    predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]  # 1=Dog, 0=Cat

    # Map predictions to filenames
    filenames = test_generator.filenames
    file_ids = [int(f.split('.')[0]) for f in filenames]  # Extract file IDs from filenames (assuming they are like 1.jpg, 2.jpg, etc.)

    # Load sampleSubmission.csv to update predictions
    sample_submission = pd.read_csv('sampleSubmission.csv')
    sample_submission['label'] = predicted_classes  # Update with model predictions

    # Save updated predictions to a CSV file
    sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
    print("Predictions saved to 'updated_sampleSubmission.csv'.")
else:
    print("No test images found in the test directory. Please check the directory structure.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
'''
'''
#new one with space issue


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np

# [Previous code remains the same until Step 7]

# Step 7: Test and Predict
def predict_in_batches(model, test_dir, batch_size=32):
    # Create a list of all test files
    test_filenames = sorted([f for f in os.listdir(test_dir) 
                           if f.endswith(('.jpg', '.jpeg'))])
    
    if len(test_filenames) == 0:
        print("No files found in the test directory.")
        return None, None
    
    # Initialize lists to store predictions and file IDs
    all_predictions = []
    file_ids = []
    
    # Process images in batches
    for i in range(0, len(test_filenames), batch_size):
        batch_files = test_filenames[i:i + batch_size]
        batch_images = []
        
        # Process each image in the current batch
        for filename in batch_files:
            img_path = os.path.join(test_dir, filename)
            img = load_img(img_path, target_size=(150, 150))
            img_array = img_to_array(img) / 255.0
            batch_images.append(img_array)
            file_ids.append(int(filename.split('.')[0]))
        
        # Convert batch to numpy array and predict
        batch_images = np.array(batch_images)
        batch_predictions = model.predict(batch_images, verbose=0)
        all_predictions.extend(batch_predictions)
        
        # Print progress
        print(f"Processed {min(i + batch_size, len(test_filenames))}/{len(test_filenames)} images")
    
    return np.array(all_predictions), file_ids

# Use the new batch processing function
print("Processing test images in batches...")
predictions, file_ids = predict_in_batches(model, test_dir)

if predictions is not None:
    # Convert predictions to binary classes
    predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]
    
    # Load and update sample submission
    sample_submission = pd.read_csv('sampleSubmission.csv')
    sample_submission['label'] = predicted_classes
    
    # Save predictions
    sample_submission.to_csv('updated_sampleSubmission.csv', index=False)
    print("Predictions saved to 'updated_sampleSubmission.csv'.")

# Save the model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")'''

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np

# Step 1: Define dataset paths
train_dir = 'data/train'
test_dir = 'data/test1'

# Step 2: Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Step 3: Build the CNN Model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create and compile the model
model = create_model()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 4: Train the Model
print("Training the model...")
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator
)

# Step 5: Evaluate the Model
print("\nEvaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Step 6: Prediction Function
def predict_in_batches(model, test_dir, batch_size=32):
    test_filenames = sorted([f for f in os.listdir(test_dir) 
                           if f.endswith(('.jpg', '.jpeg'))])
    
    if len(test_filenames) == 0:
        print("No files found in the test directory.")
        return None, None
    
    all_predictions = []
    file_ids = []
    
    print(f"\nTotal images to process: {len(test_filenames)}")
    
    for i in range(0, len(test_filenames), batch_size):
        batch_files = test_filenames[i:i + batch_size]
        batch_images = []
        
        for filename in batch_files:
            try:
                img_path = os.path.join(test_dir, filename)
                img = load_img(img_path, target_size=(150, 150))
                img_array = img_to_array(img) / 255.0
                batch_images.append(img_array)
                file_ids.append(int(filename.split('.')[0]))
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        if batch_images:  # Only process if we have images in the batch
            batch_images = np.array(batch_images)
            batch_predictions = model.predict(batch_images, verbose=0)
            all_predictions.extend(batch_predictions)
            
            print(f"Processed {min(i + batch_size, len(test_filenames))}/{len(test_filenames)} images")
    
    return np.array(all_predictions), file_ids

# Step 7: Process Test Images
print("\nProcessing test images in batches...")
predictions, file_ids = predict_in_batches(model, test_dir)

if predictions is not None:
    predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]
    
    # Create DataFrame for submission
    submission_df = pd.DataFrame({
        'id': file_ids,
        'label': predicted_classes
    })
    
    # Sort by id to ensure correct order
    submission_df = submission_df.sort_values('id')
    
    # Save predictions
    submission_df.to_csv('updated_sampleSubmission.csv', index=False)
    print("\nPredictions saved to 'updated_sampleSubmission.csv'.")

# Step 8: Save the Model
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")