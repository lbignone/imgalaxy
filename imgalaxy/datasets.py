"""Galaxy Zoo 3D Dataset."""
import tensorflow_datasets as tfds

from imgalaxy.cfg import BASE_URL, DATA_DIR

IMAGES = [
    "gz3d_1-641253_37_14744505",
    "gz3d_1-641270_127_14744506",
    "gz3d_1-82618_19_14714437",
    "gz3d_1-82618_19_14714437",
    "gz3d_1-82918_127_14714487",
]


class GalaxyZoo3Dataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    _URL = BASE_URL
    _URL_GZ = "https://data.galaxyzoo.org/"
    _DESCRIPTION = "???"
    _CITATION = "???"

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

        data_path = dl_manager.DATA_DIR
        paths = {"images_path": data_path}

        return {"train": self._generate_examples(paths)}

    def _generate_examples(self, path):
        """Generator of examples for each split."""
        for galaxy_path in path.glob('*.gz'):  # to loop over galaxies once
            name = galaxy_path.with_suffix("").with_suffix("").name  # remove .fits.gz
            yield name, {
                'image': DATA_DIR / f"{name}_image.npy",
                'mask_center': DATA_DIR / f"{name}_mask_center.npy",
                'mask_stars': DATA_DIR / f"{name}_mask_stars.npy",
                'mask_spiral': DATA_DIR / f"{name}_mask_spiral.npy",
                'mask_bar': DATA_DIR / f"{name}_mask_bar.npy",
            }
