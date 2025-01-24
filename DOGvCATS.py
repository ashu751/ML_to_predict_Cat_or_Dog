import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np


train_dir = 'data/train'
test_dir = 'data/test1'


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


model = create_model()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


print("Training the model...")
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator
)


print("\nEvaluating the model on the validation set...")
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {validation_accuracy:.2f}")


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
        
        if batch_images:  
            batch_images = np.array(batch_images)
            batch_predictions = model.predict(batch_images, verbose=0)
            all_predictions.extend(batch_predictions)
            
            print(f"Processed {min(i + batch_size, len(test_filenames))}/{len(test_filenames)} images")
    
    return np.array(all_predictions), file_ids

print("\nProcessing test images in batches...")
predictions, file_ids = predict_in_batches(model, test_dir)

if predictions is not None:
    predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]
    
    
    submission_df = pd.DataFrame({
        'id': file_ids,
        'label': predicted_classes
    })
    
    
    submission_df = submission_df.sort_values('id')
    
 
    submission_df.to_csv('updated_sampleSubmission.csv', index=False)
    print("\nPredictions saved to 'updated_sampleSubmission.csv'.")

#
model.save("dog_vs_cat_classifier.h5")
print("Model saved as 'dog_vs_cat_classifier.h5'.")
