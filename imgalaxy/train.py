import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
from wandb.keras import WandbMetricsLogger

from imgalaxy.constants import (
    BUFFER_SIZE,
    IMAGE_SIZE,
    MASK,
    MIN_VOTE,
    NUM_EPOCHS,
    THRESHOLD,
)
from imgalaxy.helpers import dice, evaluate_model, jaccard


def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    return input_image, input_mask


def binary_mask(input_mask, threshold: int = THRESHOLD):
    input_mask = tf.where(
        input_mask < threshold, tf.zeros_like(input_mask), tf.ones_like(input_mask)
    )

    return input_mask


def load_image_train(datapoint):
    input_image = datapoint['image']
    input_mask = datapoint[MASK]
    input_image = tf.image.resize(
        input_image, (IMAGE_SIZE, IMAGE_SIZE), method="nearest"
    )
    input_mask = tf.image.resize(input_mask, (IMAGE_SIZE, IMAGE_SIZE), method="nearest")
    input_image, input_mask = augment(input_image, input_mask)
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = binary_mask(input_mask, THRESHOLD)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = datapoint['image']
    input_mask = datapoint[MASK]
    input_image = tf.image.resize(
        input_image, (IMAGE_SIZE, IMAGE_SIZE), method="nearest"
    )
    input_mask = tf.image.resize(input_mask, (IMAGE_SIZE, IMAGE_SIZE), method="nearest")
    input_image = tf.cast(input_image, tf.float32) / 255.0  # normalize
    input_mask = binary_mask(input_mask, THRESHOLD)

    return input_image, input_mask


def double_conv_block(x, n_filters, batch_normalization: bool = False):
    x = layers.Conv2D(
        n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal"
    )(x)
    if batch_normalization:
        x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal"
    )(x)
    if batch_normalization:
        x = layers.BatchNormalization()(x)
    return x


def downsample_block(
    x, n_filters, dropout_rate: float = 0.3, batch_normalization: bool = False
):
    f = double_conv_block(x, n_filters, batch_normalization=batch_normalization)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout_rate)(p)

    return f, p


def upsample_block(
    x,
    conv_features,
    n_filters,
    dropout_rate: float = 0.3,
    batch_normalization: bool = False,
):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(dropout_rate)(x)
    x = double_conv_block(x, n_filters, batch_normalization=batch_normalization)

    return x


def build_unet_model(dropout_rate: float = 0.3, batch_normalization: bool = False):
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    f1, p1 = downsample_block(
        inputs,
        IMAGE_SIZE // 2,
        dropout_rate=dropout_rate,
        batch_normalization=batch_normalization,
    )
    f2, p2 = downsample_block(
        p1,
        IMAGE_SIZE,
        dropout_rate=dropout_rate,
        batch_normalization=batch_normalization,
    )
    f3, p3 = downsample_block(
        p2,
        IMAGE_SIZE * 2,
        dropout_rate=dropout_rate,
        batch_normalization=batch_normalization,
    )
    f4, p4 = downsample_block(
        p3,
        IMAGE_SIZE * 4,
        dropout_rate=dropout_rate,
        batch_normalization=batch_normalization,
    )

    bottleneck = double_conv_block(
        p4, IMAGE_SIZE * 8, batch_normalization=batch_normalization
    )

    u6 = upsample_block(
        bottleneck,
        f4,
        IMAGE_SIZE * 4,
        dropout_rate=dropout_rate,
        batch_normalization=batch_normalization,
    )
    u7 = upsample_block(
        u6,
        f3,
        IMAGE_SIZE * 2,
        dropout_rate=dropout_rate,
        batch_normalization=batch_normalization,
    )
    u8 = upsample_block(
        u7,
        f2,
        IMAGE_SIZE,
        dropout_rate=dropout_rate,
        batch_normalization=batch_normalization,
    )
    u9 = upsample_block(
        u8,
        f1,
        IMAGE_SIZE // 2,
        dropout_rate=dropout_rate,
        batch_normalization=batch_normalization,
    )

    outputs = layers.Conv2D(2, 1, padding="same", activation="softmax")(u9)

    model = tf.keras.Model(inputs, outputs, name="U-Net")

    return model


def train_pipeline(
    loss: str = "sparse_categorical_crossentropy",
    dropout_rate: float = 0.3,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = 0.0011,
    batch_size: int = 32,
):
    ds, _ = tfds.load(
        'galaxy_zoo3d', split=['train[:75%]', 'train[75%:]'], with_info=True
    )
    ds_train, ds_test = ds[0], ds[1]

    ds_train = ds_train.filter(lambda x: tf.reduce_max(x[MASK]) >= MIN_VOTE)
    ds_test = ds_test.filter(lambda x: tf.reduce_max(x[MASK]) >= MIN_VOTE)
    if MASK == 'spiral_mask':
        TRAIN_LENGTH, VAL_SIZE, TEST_SIZE = 4883, 1088, 551
    elif MASK == 'bar_mask':
        TRAIN_LENGTH, VAL_SIZE, TEST_SIZE = 3783, 832, 421

    train_dataset = ds_train.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = ds_test.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = (
        train_dataset.cache().shuffle(BUFFER_SIZE).batch(batch_size).repeat()
    )
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_batches = test_dataset.take(VAL_SIZE).batch(batch_size)
    test_batches = test_dataset.skip(VAL_SIZE).take(TEST_SIZE).batch(batch_size)
    unet_model = build_unet_model(dropout_rate)

    unet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy", jaccard, dice],
        jit_compile=True,
    )

    STEPS_PER_EPOCH = TRAIN_LENGTH // batch_size

    VAL_SUBSPLITS = 5
    TEST_LENGTH = VAL_SIZE + TEST_SIZE
    VALIDATION_STEPS = TEST_LENGTH // batch_size // VAL_SUBSPLITS
    model_history = unet_model.fit(  # pylint: disable=unused-variable
        train_batches,
        epochs=num_epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        validation_data=validation_batches,
        callbacks=[WandbMetricsLogger()],
    )

    confusion_mx, _jacc = evaluate_model(test_batches, unet_model)
    return unet_model, model_history, confusion_mx, _jacc


if __name__ == '__main__':
    # MODEL_N = 0
    # import wandb
    # wandb.init(
    #    project="galaxy-segmentation-project",
    #    name=f"jose_{MODEL_N}",
    #    config={
    #       'loss': 'cross_entropy',
    #       'dropout': 0.3,
    #       'mask': 'bar_mask',
    #       'size': IMAGE_SIZE,
    #       'threshold': THRESHOLD,
    #       'NUM_EPOCHS': NUM_EPOCHS,
    #       'group': f"jose_{MASK}",
    #    },
    # )
    unet, history, conf_mx, jacc = train_pipeline()
