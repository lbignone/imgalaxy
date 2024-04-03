import click
import wandb

from imgalaxy.constants import IMAGE_SIZE, NUM_EPOCHS, THRESHOLD
from imgalaxy.unet import UNet


@click.command()
@click.option("--dropout-rate", default=0.3, help="Inner-layer dropout rate.")
@click.option("--mask", default='bar_mask', help="Target mask to use for training.")
@click.option("--learning-rate", default=0.0011, help="Learning rate for training.")
@click.option(
    "--num-epochs", default=NUM_EPOCHS, help="Total number of epochs (if no patience)."
)
@click.option(
    "--batch-size", default=32, help="Size of the batches used for train/test."
)
@click.option(
    "--image-size", default=IMAGE_SIZE, help="Input images are resized for training."
)
@click.option(
    "--n-filters", default=128, help="Base number of filters for convolutions."
)
@click.option(
    "--min-vote", default=3, help="Min votes for a pixel to be positvely labeled."
)
@click.option(
    "--batch-normalization",
    default=False,
    help="Batch normalization in each double convolution.",
)
@click.option(
    "--loss",
    default="sparse_categorical_crossentropy",
    help="Loss function for training.",
)
def train(
    loss,
    dropout_rate,
    num_epochs,
    learning_rate,
    batch_size,
    batch_normalization,
    image_size,
    n_filters,
    mask,
    min_vote,
):
    wandb.init(
        project="galaxy-segmentation-project",
        name="jose_0",
        config={
            'loss': loss,
            'dropout': dropout_rate,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'batch_normalization': batch_normalization,
            'size': image_size,
            'n_filters': n_filters,
            'mask': mask,
            'min_vote': min_vote,
            'threshold': THRESHOLD,
            'group': f"jose_{mask}",
        },
    )

    unet = UNet(
        loss=loss,
        dropout_rate=dropout_rate,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        batch_normalization=batch_normalization,
        image_size=image_size,
        n_filters=n_filters,
        mask=mask,
        min_vote=min_vote,
    )
    history, test_data = unet.train_pipeline()
    return history, test_data


if __name__ == '__main__':
    train()  # pylint: disable=no-value-for-parameter
