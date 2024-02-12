"""Configs."""
from pathlib import Path

import importlib_resources
from decouple import AutoConfig

PKG_PATH = importlib_resources.files('imgalaxy')
REPO_ROOT = PKG_PATH.parent

config = AutoConfig(search_path=REPO_ROOT)

DEFAULT_PATH = Path(config("GALAXY_ZOO_DIR", cast=str))
RESOURCES_DIR = PKG_PATH / "resources"
DATA_DIR = RESOURCES_DIR / "dataset"
LOGS_DIR = RESOURCES_DIR / "logs"

Path(DEFAULT_PATH).mkdir(exist_ok=True, parents=True)
Path(RESOURCES_DIR).mkdir(exist_ok=True, parents=True)
Path(DATA_DIR).mkdir(exist_ok=True, parents=True)
Path(LOGS_DIR).mkdir(exist_ok=True, parents=True)

BASE_URL = 'https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0/'
MANGA_SHA1SUM = 'manga_morphology_galaxyzoo3d_v4_0_0.sha1sum'
CHECKSUMS_FILENAME = 'checksums.sha1sum'
