import tensorflow as tf

keras = tf.keras

# reformating command + shift + L
# 아래와 같은형태 -> 튜플( 리스트(배열)과 비슷  , 데이터수정불가 )
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt

# digit = train_images[13]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# my_slice = train_images[: ,15:, 14:]
# print (  my_slice.shape )

keras.layers.Dense(512, activation='relu')
