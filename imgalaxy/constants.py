"""Constants."""


RUN_FROM = 'local'
NUM_EPOCHS = 7
SIZE = 64  # Size of resized images and masks. You may have to change batch sizes.

MASK = 'spiral_mask'
TRAIN_WITH = 'only'  # "all": all imgs in training dataset. "only" for spiraled galaxies

MIN_VOTE = 3  # min of votes that the most voted pixel of a mask must have to be considered a spiral arm (barred) galaxy.

THRESHOLD = 6  # min amount of votes that a pixel must have to be clasified as a spiral arm (bar).
PATIENCE = 3
