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
    METADATA_DIR,
    METADATA_FILENAMES,
    SHA1SUM_URL,
)

logging.basicConfig(level='INFO')
logger = logging.getLogger(f"imgalaxy.{__file__}")


def get_galaxies_metadata() -> None:
    """Download and save metadata files. Takes no arguments and returns no value."""
    logger.info("Downloading image names and sha1 checksums...")
    sha1_response = requests.get(BASE_URL + SHA1SUM_URL, timeout=360)
    with open(METADATA_DIR / CHECKSUMS_FILENAME, mode='wb') as f:
        f.write(sha1_response.content)
    logger.info(f"Checksums file saved in {METADATA_DIR / CHECKSUMS_FILENAME}.")

    for name in METADATA_FILENAMES[1:]:
        logger.info(f"Downloading {name.replace('.fits', '')}.")
        response = requests.get(BASE_URL + name, timeout=180)
        with open(METADATA_DIR / name, mode='wb') as f:
            f.write(response.content)
            logger.info(f"File {METADATA_DIR / name} saved.")


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


def save_galaxy_npy(filepath: Path) -> None:
    """Save numpy representation of a galaxy's RGB image and its masks.

    Parameters
    ----------
    filepath : pathlib.Path
        Full path for the galaxy's file (.fits compatible format)

    Returns no value.

    """
    location = filepath.parent
    name = filepath.with_suffix('').with_suffix('')  # rm .fits & .gz from filename
    with fits.open(filepath) as hdul:
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
    filepath = DATA_DIR / galaxy_filename
    if galaxy_filename.endswith(".fits"):
        filepath = METADATA_DIR / galaxy_filename
        save_npy = False  # these files were already downloaded by now.
    with open(filepath, 'wb') as f:
        f.write(response.content)
        if save_npy:
            save_galaxy_npy(filepath)

    if checksum:
        image_sha = str(galaxy).split(' ')[0]
        if not verify_checksum(filepath, image_sha):
            raise ValueError(f"Wrong sha1 values for galaxy {galaxy}.")


def download_pipeline(
    start: int = 0, verify_checksums: bool = True, save_npy: bool = True
):
    """Download galaxies from GZ3D dataset.

    Parameters
    ----------
    start : int, between 0 and 29815, default=0.
        Starting point of the download. To not start over in the case of an interruption.
    verify_checksums : bool, default=True
        Verify checksum for downloaded images.
    save_npy : bool, default=True
        Save galaxies' image and masks as .npy numpy arrays.

    Returns no value.

    """
    get_galaxies_metadata()
    galaxies_metadata = open(METADATA_DIR / CHECKSUMS_FILENAME, 'r').readlines()[start:]
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
    help="Save copies of as .npy (numpy arrays).",
)
def cli(start, verify_checksums, save_npy):
    """CLI wrapper around `download_galaxy()`."""
    get_galaxies_metadata()
    galaxies_metadata = open(METADATA_DIR / CHECKSUMS_FILENAME, 'r').readlines()[start:]
    logger.info("Downloading Galaxy Zoo 3D dataset. This may take several hours...")
    with click.progressbar(
        galaxies_metadata,
        empty_char="☆",
        fill_char="★",
        width=0,
        length=len(galaxies_metadata),
        show_pos=True,
    ) as galaxies:
        for galaxy in galaxies:
            download_galaxy(galaxy, checksum=verify_checksums, save_npy=save_npy)
    logger.info("All done, bye =)")


if __name__ == '__main__':
    cli()  # pylint: disable = no-value-for-parameter
