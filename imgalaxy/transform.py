"""Transformation methods."""
import tensorflow as tf
import tensorflow_datasets as tfds
from astropy.io import fits

from imgalaxy.cfg import DEFAULT_PATH

IMAGES = [
    "gz3d_1-641253_37_14744505",
    "gz3d_1-641270_127_14744506",
    "gz3d_1-82618_19_14714437",
    "gz3d_1-82618_19_14714437",
    "gz3d_1-82918_127_14714487",
]


if __name__ == '__main__':
    with fits.open(f'/hdd/galaxy-zoo/mapas/{IMAGES[0]}.fits.gz') as hdul:
        _data = hdul[1].data.copy()
        dataset = tf.data.Dataset(_data)  # pylint: disable=abstract-class-instantiated

    ds, info = tfds.load(
        'galaxy_zoo3d',
        data_dir=DEFAULT_PATH,
        split=['train[:75%]', 'train[75%:]'],
        with_info=True,
    )
