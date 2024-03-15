import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from keras import layers
from sklearn.metrics import confusion_matrix, jaccard_score
from wandb.keras import WandbMetricsLogger

from imgalaxy.constants import MASK, MIN_VOTE, NUM_EPOCHS, SIZE, THRESHOLD, TRAIN_WITH

# from galaxies_datasets import datasets


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
    input_image = tf.image.resize(input_image, (SIZE, SIZE), method="nearest")
    input_mask = tf.image.resize(input_mask, (SIZE, SIZE), method="nearest")
    input_image, input_mask = augment(input_image, input_mask)
    input_image = tf.cast(input_image, tf.float32) / 255.0  # normalize
    input_mask = binary_mask(input_mask, THRESHOLD)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = datapoint['image']
    input_mask = datapoint[MASK]
    input_image = tf.image.resize(input_image, (SIZE, SIZE), method="nearest")
    input_mask = tf.image.resize(input_mask, (SIZE, SIZE), method="nearest")
    input_image = tf.cast(input_image, tf.float32) / 255.0  # normalize
    input_mask = binary_mask(input_mask, THRESHOLD)

    return input_image, input_mask


def double_conv_block(x, n_filters):
    x = layers.Conv2D(
        n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal"
    )(x)
    x = layers.Conv2D(
        n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal"
    )(x)

    return x


def downsample_block(x, n_filters, dropout_rate: float = 0.3):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout_rate)(p)

    return f, p


def upsample_block(x, conv_features, n_filters, dropout_rate: float = 0.3):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(dropout_rate)(x)
    x = double_conv_block(x, n_filters)

    return x


def build_unet_model(dropout_rate: float = 0.3):
    inputs = layers.Input(shape=(SIZE, SIZE, 3))

    f1, p1 = downsample_block(inputs, SIZE // 2, dropout_rate=dropout_rate)
    f2, p2 = downsample_block(p1, SIZE, dropout_rate=dropout_rate)
    f3, p3 = downsample_block(p2, SIZE * 2, dropout_rate=dropout_rate)
    f4, p4 = downsample_block(p3, SIZE * 4, dropout_rate=dropout_rate)

    bottleneck = double_conv_block(p4, SIZE * 8)

    u6 = upsample_block(bottleneck, f4, SIZE * 4, dropout_rate=dropout_rate)
    u7 = upsample_block(u6, f3, SIZE * 2, dropout_rate=dropout_rate)
    u8 = upsample_block(u7, f2, SIZE, dropout_rate=dropout_rate)
    u9 = upsample_block(u8, f1, SIZE // 2, dropout_rate=dropout_rate)

    outputs = layers.Conv2D(2, 1, padding="same", activation="softmax")(u9)

    model = tf.keras.Model(inputs, outputs, name="U-Net")  # pylint: disable=no-member

    return model


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def evaluate_model(dataset, model, num=1):
    # TODO: try tf.keras.Model.evaluate()
    if dataset:
        for image, mask in dataset:
            pred_mask = create_mask(model.predict(image))
            for ind in range(num):
                if np.amax(pred_mask[ind].numpy()) == 0:
                    print(2 * '\n')
                    continue
                else:
                    conf_matrix = confusion_matrix(
                        pred_mask[ind].numpy().reshape(-1),
                        mask[ind].numpy().reshape(-1),
                    )
                    jacc_score = jaccard_score(
                        pred_mask[ind].numpy().reshape(-1),
                        mask[ind].numpy().reshape(-1),
                    )
        return conf_matrix, jacc_score


def train_pipeline(
    loss: str = "sparse_categorical_crossentropy",
    dropout_rate: float = 0.3,
    num_epochs: int = NUM_EPOCHS,
):
    ds, _ = tfds.load(
        'galaxy_zoo3d', split=['train[:75%]', 'train[75%:]'], with_info=True
    )
    ds_train, ds_test = ds[0], ds[1]

    if TRAIN_WITH == 'all':
        BUFFER_SIZE, BATCH_SIZE = 1000, 64
        TRAIN_LENGTH, VAL_SIZE, TEST_SIZE = 22360, 4992, 2461
    elif TRAIN_WITH == 'only':
        BUFFER_SIZE, BATCH_SIZE = 300, 32
        ds_train = ds_train.filter(lambda x: tf.reduce_max(x[MASK]) >= MIN_VOTE)
        ds_test = ds_test.filter(lambda x: tf.reduce_max(x[MASK]) >= MIN_VOTE)
        if MASK == 'spiral_mask':
            TRAIN_LENGTH, VAL_SIZE, TEST_SIZE = 4883, 1088, 551
        elif MASK == 'bar_mask':
            TRAIN_LENGTH, VAL_SIZE, TEST_SIZE = 3783, 832, 421

    train_dataset = ds_train.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = ds_test.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = (
        train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    )
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_batches = test_dataset.take(VAL_SIZE).batch(BATCH_SIZE)
    test_batches = test_dataset.skip(VAL_SIZE).take(TEST_SIZE).batch(BATCH_SIZE)

    unet_model = build_unet_model(dropout_rate)

    unet_model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # pylint: disable=no-member
        loss=loss,
        metrics=["accuracy"],
        jit_compile=True,
    )

    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    VAL_SUBSPLITS = 5
    TEST_LENGTH = VAL_SIZE + TEST_SIZE
    VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE // VAL_SUBSPLITS

    model_history = unet_model.fit(  # pylint: disable=unused-variable
        train_batches,
        epochs=num_epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        validation_data=validation_batches,
        callbacks=[WandbMetricsLogger()],
    )

    confusion_mx, jacc = evaluate_model(test_batches, unet_model)
    return unet_model, model_history, confusion_mx, jacc


if __name__ == '__main__':
    MODEL_N = 0
    wandb.init(
        project="galaxy-segmentation-project",
        name=f"jose_{MODEL_N}",
        config={
            'loss': 'cross_entropy',
            'dropout': 0.3,
            'mask': 'bar_mask',
            'size': SIZE,
            'threshold': THRESHOLD,
            'NUM_EPOCHS': NUM_EPOCHS,
        },
    )
    unet, history, conf_mx, jaccard = train_pipeline()
    wandb.log({"confusion_mx": conf_mx, "jaccard": jaccard})
