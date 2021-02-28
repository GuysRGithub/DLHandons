from IPython.display import Image as IImage, display
import numpy as np
import PIL
from PIL import Image
import random
import requests
import tensorflow as tf


d = requests.get("https://www.paristoolkit.com/Images/xeffel_view.jpg.pagespeed.ic.8XcZNqpzSj.jpg")
with open("image.jpeg", "wb") as f:
    f.write(d.content)

img = PIL.Image.open("image.jpeg")
img.load()
img_array = np.array(img)
PIL.Image.fromarray(img_array)


def random_flip_right(image):
    return tf.image.random_flip_left_right(image)


def random_contrast(image, minval=.6, maxval=1.4):
    r = tf.random.uniform([], minval=minval, maxval=maxval)
    image = tf.image.adjust_contrast(image, contrast_factor=r)
    return tf.cast(image, tf.uint8)


def random_saturation(image, minval=0.4, maxval=2.):
    r = tf.random.uniform([], minval=minval, maxval=maxval)
    image = tf.image.adjust_saturation(image, r)
    return tf.cast(image, tf.uint8)


def random_hue(image, minval=-0.04, maxval=.08):
    r = tf.random.uniform([], minval=minval, maxval=maxval)
    image = tf.image.adjust_hue(image, delta=r)
    return tf.cast(image, tf.uint8)


def distorted_random_crop(image, min_object_covered=0.1,
                          aspect_ratio_range=(3. / 4., 4. / 3.),
                          area_range=(0.06, 1.0),
                          max_attempts=100,
                          scope=None):
    crop_box = tf.constant([.0, .0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    distorted_bounding_box = \
        tf.image.sample_distorted_bounding_box(tf.shape(image),
                                               bounding_boxes=crop_box,
                                               min_object_covered=min_object_covered,
                                               aspect_ratio_range=aspect_ratio_range,
                                               area_range=area_range,
                                               max_attempts=max_attempts,
                                               use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = distorted_bounding_box

    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image


def resize_image(image):
    image = tf.image.resize(image, size=(256, 256), preserve_aspect_ratio=False)
    return tf.cast(image, tf.uint8)


PIL.Image.fromarray(resize_image(img_array).numpy()).show()
