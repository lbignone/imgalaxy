"""Transformation methods."""
import tensorflow as tf
import tensorflow_datasets as tfds


def resize(input_image, input_mask, size):
    input_image = tf.image.resize(input_image, (size, size), method="nearest")
    input_mask = tf.image.resize(input_mask, (size, size), method="nearest")

    return input_image, input_mask


def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    return input_image, input_mask


def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0

    return input_image


def binary_mask(input_mask, threshold):
    input_mask = tf.where(
        input_mask < threshold, tf.zeros_like(input_mask), tf.ones_like(input_mask)
    )

    return input_mask


if __name__ == '__main__':
    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    breakpoint()
    ds, info = tfds.load(
        'galaxy_zoo3d', split=['train[:75%]', 'train[75%:]'], with_info=True
    )
