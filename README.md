import os
import datetime
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121  # Using DenseNet121 for this example

def create_densenet_model(num_classes):
    base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # Set the DenseNet layers to be trainable for fine-tuning
    for layer in base_model.layers:
        layer.trainable = True
    return model

# Paths for the dataset
base_dir = r'E:\Picture\Data_Image1\Dataset_For_Grading_CLAHE'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Data generators with preprocessing function for DenseNet
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)

# Data generators
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# Number of classes
num_classes = len(train_generator.class_indices)

# Create and compile the DenseNet model
model = create_densenet_model(num_classes)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Compute class weights for imbalanced datasets
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights_dict = dict(enumerate(class_weights))

# Callbacks for saving the model and reducing learning rate
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath='best_model_DenseNet.h5', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.00001),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model_path = os.path.join(base_dir, 'model', 'best_model_DenseNet.h5')
model.save(model_path)

# Save the class indices
class_indices = train_generator.class_indices
class_indices_path = os.path.join(base_dir, 'model', 'class_indices_DenseNet.json')
with open(class_indices_path, 'w') as json_file:
    json.dump(class_indices, json_file)

# Plot and display training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

print("Model and class indices have been successfully saved.")
