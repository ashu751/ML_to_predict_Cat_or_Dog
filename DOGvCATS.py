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
