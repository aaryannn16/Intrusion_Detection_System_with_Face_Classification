import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Load pre-trained ResNet50V2 model without the top classification layer
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new layers on top of the pre-trained base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pooling to reduce dimensionality

# Add dense layers with batch normalization and dropout
x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Output layer for 3 classes (face classification)
predictions = Dense(3, activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tuning: Initially freeze all layers, then unfreeze progressively
for layer in base_model.layers:
    layer.trainable = False
model.trainable = True

# Compile the model with a better optimizer and loss function
model.compile(optimizer=AdamW(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, zoom_range=0.3,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
    horizontal_flip=True, fill_mode='nearest'
)

# Data preparation for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Training and validation data generators
train_generator = train_datagen.flow_from_directory('MP Dataset', target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory('MP Dataset', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Callbacks: Learning rate scheduler, early stopping, and model checkpoint
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('face_classification_model_fixed.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model
history = model.fit(train_generator, epochs=30, validation_data=validation_generator, callbacks=[early_stopping, lr_scheduler, model_checkpoint])

# Fine-tune by unfreezing more layers and reducing the learning rate
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=AdamW(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning training
history_fine = model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks=[early_stopping, lr_scheduler, model_checkpoint])

# Save the fine-tuned model
model.save('fine_tuned_face_classification_model.h5')