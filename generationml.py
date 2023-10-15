import opendatasets as od
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.layers import Dense, Flatten
from keras.models import Model
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Function to define the prediction
def prediction(path):
    img = load_img(path, target_size=(256, 256))
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis=0)
    pred = np.argmax(loaded_model.predict(img))
    return ref[pred]

# Download your dataset if not already done
# od.download("new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)")
def prediction11():
  len(os.listdir('new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'))

  train_datagen = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, preprocessing_function=preprocess_input, horizontal_flip=True)
  val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

  train = train_datagen.flow_from_directory(directory='new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
                                            target_size=(256, 256),
                                            batch_size=32)

  val = val_datagen.flow_from_directory(directory='new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
                                        target_size=(256, 256),
                                        batch_size=32)

  t_img, label = train.next()

  def plotImage(img_arr, label):
      for im, l in zip(img_arr, label):
          plt.figure(figsize=(5, 5))
          plt.imshow(im)
          plt.show

  plotImage(t_img[:3], label[3:])

  base_model = VGG19(input_shape=(256, 256, 3), include_top=False)

  for layer in base_model.layers:
      layer.trainable = False

  X = Flatten()(base_model.output)
  X = Dense(units=38, activation='softmax')(X)

  model = Model(base_model.input, X)

  model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

  es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=3, verbose=1)
  mc = ModelCheckpoint(filepath="best_model.h5", monitor="val_accuracy", min_delta=0.01, patience=3, verbose=1, save_best_only=True)

  cb = [es, mc]

  his = model.fit_generator(train, steps_per_epoch=16, epochs=1,
                            verbose=1, validation_data=val, validation_steps=16)

  # Save the trained model to a file
  model.save("plant_disease_model.h5")

  # Create a dictionary to map class indices to class names
  ref = dict(zip(list(train.class_indices.values()) , list(train.class_indices.keys())))

  # Load the saved model
  from keras.models import load_model

  # Load the saved model
  loaded_model = load_model("plant_disease_model.h5")

  # Example usage of the prediction function:
