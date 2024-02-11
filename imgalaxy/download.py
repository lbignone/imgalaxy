import logging
from hashlib import sha1
from pathlib import Path
from urllib.request import urlretrieve

import click

from imgalaxy.cfg import BASE_URL, CHECKSUMS_FILENAME, DEFAULT_PATH, MANGA_SHA1SUM

logging.basicConfig(level='INFO')
logger = logging.getLogger(f"imgalaxy.{__file__}")


def get_files_and_sha1sum(filepath: str = DEFAULT_PATH):
    """Retrieve images' names and checksums and save them at `filepath`.

    Parameters
    ----------
    filepath : str, default=`DEFAULT_PATH@cfg.py`.
        Location to save the checksums file.

    Returns
    -------
    str
        Location of the downloaded file.

    """
    path = Path(filepath)
    if path.is_dir():
        logger.info(f"Downloading checksums file to {path}.")
    else:
        logger.error(f"Location {filepath} not found. Creating it..")
        Path(path).mkdir(parents=True, exist_ok=True)
    urlretrieve(BASE_URL + MANGA_SHA1SUM, Path(filepath) / CHECKSUMS_FILENAME)
    return path / CHECKSUMS_FILENAME


def verify_checksum(filepath, sha1_hash):
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


@click.command()
@click.option(
    "--path", "-p", required=False, default=DEFAULT_PATH, help="Location to save data."
)
@click.option(
    "--start", "-s", required=False, default=0, help="Starting point of download."
)
def main(path, start):
    path = Path(path)
    sha1sums_filepath = get_files_and_sha1sum(path)
    sha1sums = open(sha1sums_filepath, 'r').readlines()[start:]
    failed_checksums: dict = {}
    with click.progressbar(sha1sums, empty_char='☆', fill_char='★', width=87) as bar:
        for image in bar:
            image_filename = str(image).split(' ')[2]
            image_sha = str(image).split(' ')[0]
            image_filepath = path / image_filename
            urlretrieve(BASE_URL + image_filename, image_filepath)
            if not verify_checksum(image_filepath, image_sha):
                failed_checksums[image_filename] = image_sha


if __name__ == '__main__':
    main()  # pylint: disable = no-value-for-parameter
