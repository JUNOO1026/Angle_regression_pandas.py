import os, glob, sys
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from pathlib import Path    #path를 객체 처럼 사용(즉, 디렉토리를 왔다갔다 할 수 있음.)
import tensorflow as tf
import pandas as pd
from keras.models import load_model
np.set_printoptions(threshold=sys.maxsize)

print('Python version', sys.version)
print('Tensorflow version', tf.__version__)
print('keras version', keras.__version__)

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

path = os.getcwd()
print(path)

folder_directory_path = './image_data_canny_blur/'

image_dir = Path(folder_directory_path)

filepaths = pd.Series(list(image_dir.glob(r'**/*.bmp')), name='Filepath').astype(str)
print(filepaths)

print(filepaths.values)
length = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='length').astype(np.int)

print(type(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]).astype(np.int)))
# print(os.path.split(os.path.split(filepaths.values[0])[0])[1])

print("##########################")
print(length)

images = pd.concat([filepaths, length], axis=1)

print(images)

train_df, test_df = train_test_split(images, train_size=0.8, test_size=0.2, shuffle=True, random_state=1)

print(type(train_df))
# print(train_df.columns)
# print(train_df.shape)
# print(type(test_df))
# print(test_df.shape)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='length',
    target_size=(160, 45),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    seed=42,
    shuffle=True,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='length',
    target_size=(160, 45),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    seed=42,
    shuffle=True,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='length',
    target_size=(160, 45),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)


print(type(train_images))
print(type(val_images))
print(type(test_images))

inputs = tf.keras.Input(shape=(160, 45, 1))
# print(inputs)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)   # input단
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)   # hidden layer 1단
# x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(inputs)   # hidden layer 1단
x = tf.keras.layers.MaxPooling2D()(x)
print(x.shape)
# x = tf.keras.layers.Dropout(rate=0.3)(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(x)       # hidden layer 2단
x = tf.keras.layers.MaxPooling2D()(x)
print(x.shape)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)       # hidden layer 3단
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(x)       # hidden layer 3단
x = tf.keras.layers.Dropout(rate=0.2)(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(x)       # hidden layer 3단
x = tf.keras.layers.MaxPooling2D()(x)
# x = tf.keras.layers.Conv2D(filters=512, kernel_size=(5, 5), activation='relu')(x)       # hidden layer 4단
# x = tf.keras.layers.MaxPooling2D()(x)
# x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation='relu')(x)       # hidden layer 5단
# x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)                                         # flatten 과정

x = tf.keras.layers.Dense(1000, activation='relu')(x)                                   # hidden layer 6단
outputs = tf.keras.layers.Dense(1, activation='linear')(x)                              # output 단

# 지금까지 문제였던 것이 과적합이 아니라 필터가 적고, 학습량이 적어서 발생했던 문제였던 듯.
# 좀 더 높은 정확값을 얻고 싶으면 보다 더 hidden layer 층을 늘리고 학습량을 늘리면 될 것 같음.

angle_model = tf.keras.Model(inputs=inputs, outputs=outputs)

angle_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

angle_model.summary()

# print(type(train_images))
#
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=200,
    # callbacks=[
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor='val_loss',
    #         patience=5,
    #         restore_best_weights=True
    # )]
)

predicted_length = np.squeeze(model.predict(test_images))
true_angle = test_images.labels

print(true_angle)
print((model.predict(test_images)))

model.save('atonomous_vehicle_regression_Angle_pandas_test1234.h5')
# print('여기서부터 예측을 한다.!!!!!!!!!!!!!!!')
# print('######################################')
#
#
# # print(np.mean(true_length[5]))
#
