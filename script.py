import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('fire_1.h5')

print(model.layers[0].input_shape)

image_path = 'input.jpg'
img = tf.keras.utils.load_img(
    image_path,
    grayscale=False,
    color_mode='rgb',
    target_size=(224, 224)
)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
result = model.predict(img)
