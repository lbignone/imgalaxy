"""Galaxy Zoo 3D Dataset."""
# import tensorflow as tf
import tensorflow_datasets as tfds
from astropy.io import fits

from imgalaxy.cfg import DATA_DIR

IMAGES = [
    "gz3d_1-641253_37_14744505",
    "gz3d_1-641270_127_14744506",
    "gz3d_1-82618_19_14714437",
    "gz3d_1-82618_19_14714437",
    "gz3d_1-82918_127_14714487",
]


class GalaxyZoo3Dataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version('0.1.0')
    RELEASE_NOTES = {'0.1.0': 'Initial release.'}

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    'image': tfds.features.Image(shape=(525, 525, 3)),
                    'mask_spiral': tfds.features.Image(shape=(525, 525, 1)),
                    'mask_bar': tfds.features.Image(shape=(525, 525, 1)),
                    'mask_center': tfds.features.Image(shape=(525, 525, 1)),
                    'mask_stars': tfds.features.Image(shape=(525, 525, 1)),
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits."""
        datasets_path = DATA_DIR / 'dataset'  # pylint: disable=unused-variable # noqa

        data_path = dl_manager.datasets_path
        paths = {"images_path": data_path}

        return {"train": self._generate_examples(paths)}

    def _generate_examples(self, path):
        """Generator of examples for each split."""
        for img_path in path.glob('*.gz'):
            with fits.open(img_path) as hdul:
                # pylint: disable=no-member
                yield img_path.name, {
                    'image': hdul[0].data.copy(),
                    'mask_center': hdul[1].data.copy(),
                    'mask_stars': hdul[2].data.copy(),
                    'mask_spiral': hdul[3].data.copy(),
                    'mask_bar': hdul[4].data.copy(),
                }
