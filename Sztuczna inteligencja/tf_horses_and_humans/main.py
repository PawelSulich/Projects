#from data_seperator import data_seperation
#data_seperation()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop

train_horse_dir = os.path.join('horse-or-human/horses')
train_human_dir = os.path.join('horse-or-human/humans')

test_horse_dir = os.path.join('test_data/horses')
test_human_dir = os.path.join('test_data/humans')
ncol = 4
nrows = 4

pic_index = 0
train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

test_horse_names = os.listdir(test_horse_dir)
test_human_names = os.listdir(test_human_dir)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index - 8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index - 8:pic_index]]

fig = plt.gcf()
fig.set_size_inches(ncol * 4, nrows * 4)
for i, img_path in enumerate(next_horse_pix + next_human_pix):
    sp = plt.subplot(nrows, ncol, i + 1)
    sp.axis('off')
    img = mpimg.imread(img_path)
    plt.imshow(img)

model = tf.keras.models.Sequential([
    Conv2D(64, (3, 3), activation="relu", input_shape=(300, 300, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(1024, activation="relu"),
    Dense(1, activation="softmax")
])
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1 / 255)
test_datagen = ImageDataGenerator(rescale = 1/255)
train_generator = train_datagen.flow_from_directory(
    'horse-or-human/',
    target_size=(300, 300),
    batch_size=70,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    'test_data/',
    target_size=(300, 300),
    batch_size=70,
    class_mode='binary'
)
history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=15,
    verbose=1,
    validation_data = test_generator,
)
from numpy.matrixlib.defmatrix import N
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():

  # predicting images
  path = '/content/' + fn
  img = load_img(path, target_size=(300, 300))
  x = img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0] > 0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")

