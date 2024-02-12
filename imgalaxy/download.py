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
    MANGA_SHA1SUM,
    RESOURCES_DIR,
)

logging.basicConfig(level='INFO')
logger = logging.getLogger(f"imgalaxy.{__file__}")


def get_galaxies_metadata() -> Path:
    """Get filenames and their checksums for the galaxies. Takes no arguments.

    Returns
    path : pathlib.Path
        Location where the metadata was saved.

    """
    path = RESOURCES_DIR / CHECKSUMS_FILENAME
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


def save_galaxy_npy(galaxy_filepath: Path) -> None:
    """Save numpy representation of a galaxy's RGB image and its masks.

    Parameters
    ----------
    galaxy_filepath : pathlib.Path
        Full path for the galaxy's file (.fits compatible format)

    Returns no value.

    """
    location = galaxy_filepath.parent
    name = galaxy_filepath.with_suffix('').stem  # remove .fits & .gz from filename
    with fits.open(galaxy_filepath) as hdul:
        # pylint: disable=no-member
        np.save(location / f"{name}_image.npy", hdul[0].data)
        np.save(location / f"{name}_mask_center.npy", hdul[1].data)
        np.save(location / f"{name}_mask_stars.npy", hdul[2].data)
        np.save(location / f"{name}_mask_spiral.npy", hdul[3].data)
        np.save(location / f"{name}_mask_bar.npy", hdul[4].data)


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


def download_pipeline(
    start: int = 0, verify_checksums: bool = True, save_npy: bool = True
):
    """Download galaxies from GZ3D dataset.

    Parameters
    ----------
    start : int, between 0 and 29815, default=0.
        Starting point of the download.
    verify_checksums : bool, default=True
        Verify checksum for downloaded images.
    save_npy : bool, default=True
        Save galaxies' image and masks as .npy numpy arrays.

    Returns no value.

    """
    logger.info("Downloading image names and sha1 checksums...")
    metadata_filepath = get_galaxies_metadata()
    galaxies_metadata = open(metadata_filepath, 'r').readlines()[start:]
    logger.info(f"Downloading from galaxy number {start}. This may take a while...")
    counter = 1
    for galaxy in galaxies_metadata:
        logger.info(f"Downloading galaxy {counter} of {len(galaxies_metadata)}.")
        download_galaxy(galaxy, checksum=verify_checksums, save_npy=save_npy)
        counter += 1


@click.command()
@click.option(
    "--start", "-s", default=0, show_default=True, help="Starting point of download."
)
@click.option(
    "--verify_checksums",
    default=True,
    show_default=True,
    help="Verify files' hash checksums.",
)
@click.option(
    "--save-npy",
    default=False,
    is_flag=True,
    show_default=True,
    help="Save copies of the images as .npy (numpy arrays).",
)
def cli(start, verify_checksums, save_npy):
    """CLI wrapper around `download_galaxy()`."""
    logger.info("Downloading image names and sha1 checksums...")
    metadata_filepath = get_galaxies_metadata()
    galaxies_metadata = open(metadata_filepath, 'r').readlines()[start:]
    logger.info("Downloading Galaxy Zoo 3D dataset. This may take several hours...")
    with click.progressbar(
        galaxies_metadata,
        empty_char="☆",
        fill_char="★",
        width=0,
        length=len(galaxies_metadata),
        show_pos=True,
    ) as count:
        for galaxy in count:
            download_galaxy(galaxy, checksum=verify_checksums, save_npy=save_npy)


if __name__ == '__main__':
    cli()  # pylint: disable = no-value-for-parameter
