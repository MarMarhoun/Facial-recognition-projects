# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 00:34:33 2021

@author: Marmarhoun
"""

# import libraries
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
print("Done!")

###### 1) Upload dataset from directories
DATADIR = "./data/train/"
CATEGORIES = ['female', 'male']

IMG_SIZE = 256
batch_size= 32 # Others batch sizes 64, 128, and 256.


ds_train_ = tf.keras.preprocessing.image_dataset_from_directory(DATADIR,
    labels="inferred",
    label_mode = "binary", # int or categorical
    class_names = CATEGORIES,
    #color_mode ="grayscale", # Uncomment thi to wirk in the gray scale
    batch_size = batch_size,
    image_size = (IMG_SIZE,IMG_SIZE), # reshape if not in this size
    shuffle = True,
    seed= 123, # to maintain the same data when spliyying the data
    validation_split = 0.2,
    subset = "training",
)


ds_valid_ = tf.keras.preprocessing.image_dataset_from_directory(DATADIR,
    labels="inferred",
    label_mode = "binary", # Other label modes are: int or categorical
    class_names = CATEGORIES,
    #color_mode ="grayscale", # Uncomment thi to wirk in the gray scale
    batch_size = batch_size,
    image_size = (IMG_SIZE,IMG_SIZE), # reshape if not in this size
    shuffle = True,
    seed= 123, # to maintain the same results for different runs and data when splitting the data 
    validation_split = 0.2,
    subset = "validation",
)

print("Done!")

########### Preprocessing the data and create checkpoints
# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE # Search for this 
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# save checkpoints during training

checkpoint_path = "./model/Pretrained model/checkpoints_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

print("Done!")

######## Create the model 


model = keras.Sequential([

    # First Convolutional Block
    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  # give the input dimensions in the first layer
                  # [height, width, color channels(RGB)]
                  input_shape=[IMG_SIZE,IMG_SIZE, 3]), # Please recheck the number of channels in your input images
    layers.MaxPool2D(),

    # Second Convolutional Block
    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    # Third Convolutional Block
    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    # Classifier Head
    layers.Flatten(),
    layers.Dense(units=128, activation="relu"),
    layers.Dense(units=1, activation="sigmoid"),
])
    
model.summary()


####### Compile and fit the model

#learning_rate=0.001 
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.05, epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

print("Compile the model is done!")

### Train the model 

epochs = 3 # then use 25 and 50

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    batch_size= batch_size,  # Others batch sizes 64, 128, and 256.
    epochs=epochs,
    callbacks=[cp_callback]  # Pass callback to training
)

print("Congratualation the model learned!")