import logging
from hashlib import sha1
from pathlib import Path

import click
import numpy as np
import requests
from astropy.io import fits

from imgalaxy.cfg import (
    BASE_URL,
    CHECKSUMS_FILENAME,
    DATA_DIR,
    DEFAULT_PATH,
    MANGA_SHA1SUM,
)

logging.basicConfig(level='INFO')
logger = logging.getLogger(f"imgalaxy.{__file__}")


def get_files_and_sha1sum(filepath: Path = DEFAULT_PATH) -> Path:
    """Retrieve images' names and checksums and save them at `filepath`.

    Parameters
    ----------
    filepath : str, default=`DEFAULT_PATH@cfg.py`.
        Location to save the checksums file.

    Returns
    -------
    pathlib.Path
        Location of the downloaded file.

    """
    path = filepath / CHECKSUMS_FILENAME
    logger.info("Downloading image names and sha1 checksums...")
    response = requests.get(BASE_URL + MANGA_SHA1SUM, timeout=360)
    with open(path, mode='wb') as f:
        f.write(response.content)
    logging.info(f"Checksums file saved in {path}.")
    return path


def verify_checksum(filepath: Path, sha1_hash: str) -> bool:
    """Verifiy that the image from `filepath` has the correct sha1 hash.

    Parameters
    ----------
    filepath
        Filepath for the image.
    sha1_hash
        Value of sha1 for the file.

    Returns
    bool
        True if the image in `filepath` has the correct hash, False otherwise.

    """
    image_hash = sha1(open(filepath, 'rb').read()).hexdigest()  # nosec
    return image_hash == sha1_hash


def save_galaxy_npy(galaxy: str, location: Path = DATA_DIR) -> None:
    """Save numpy representation of a galaxy's RGB image and its masks.

    Parameters
    ----------
    galaxy : str
        Galaxy's filename.
    location : pathlib.Path, default=`DATA_DIR@cfg.py`
        Location to save binaries.

    Returns no value.

    """
    with fits.open(location / galaxy) as hdul:
        # pylint: disable=no-member
        galaxy_name = galaxy.replace(".fits.gz", "")
        np.save(location / f"{galaxy_name}_image", hdul[0].data)
        np.save(location / f"{galaxy_name}_center_mask", hdul[1].data)
        np.save(location / f"{galaxy_name}_stars_mask", hdul[2].data)
        np.save(location / f"{galaxy_name}_spiral_mask", hdul[3].data)
        np.save(location / f"{galaxy_name}_bar_mask", hdul[4].data)


def download_galaxy(galaxy: str, checksum: bool = True, save_npy: bool = False) -> None:
    """Download a single galaxy's image and masks from its filename.

    Parameters
    ----------
    galaxy : str
        Filename of the galaxy.
    checksum : bool, default=True
        Verify sha1sum values for the file.
    save_numpy : bool, default=False
        Save numpy representation of the galaxy's images using `save_galaxy_npy`.

    Returns no value.

    """
    galaxy_filename = str(galaxy).split(' ')[2].strip()
    response = requests.get(BASE_URL + galaxy_filename, timeout=180)
    galaxy_filepath = DATA_DIR / galaxy_filename
    with open(galaxy_filepath, 'wb') as f:
        f.write(response.content)

    if save_npy:
        save_galaxy_npy(galaxy_filepath)

    if checksum:
        image_sha = str(galaxy).split(' ')[0]
        if not verify_checksum(galaxy_filepath, image_sha):
            raise ValueError(f"Wrong sha1 values for galaxy {galaxy}.")


@click.command()
@click.option(
    "--path", "-p", required=False, default=DEFAULT_PATH, help="Location to save data."
)
@click.option(
    "--start", "-s", required=False, default=0, help="Starting point of download."
)
@click.option(
    "--verify-checksums", required=False, default=True, help="Disable hash checksums."
)
@click.option(
    "--save-npy",
    required=False,
    default=True,
    help="Save copies of the images as numpy arrays.",
)
def download_pipeline(path, start, verify_checksums, save_npy):
    path = Path(path)
    sha1sums_filepath = get_files_and_sha1sum(path)
    sha1sums = open(sha1sums_filepath, 'r').readlines()[start:]
    with click.progressbar(sha1sums, empty_char='☆', fill_char='★', width=87) as count:
        for galaxy in count:
            download_galaxy(galaxy, checksum=verify_checksums, save_npy=save_npy)


if __name__ == '__main__':
    download_pipeline()  # pylint: disable = no-value-for-parameter
