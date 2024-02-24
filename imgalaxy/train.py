import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers

# from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
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


def binary_mask(input_mask, threshold):
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


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


def upsample_block(x, conv_features, n_filters):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)

    return x


def build_unet_model():
    inputs = layers.Input(shape=(SIZE, SIZE, 3))

    f1, p1 = downsample_block(inputs, SIZE / 2)
    f2, p2 = downsample_block(p1, SIZE)
    f3, p3 = downsample_block(p2, SIZE * 2)
    f4, p4 = downsample_block(p3, SIZE * 4)

    bottleneck = double_conv_block(p4, SIZE * 8)

    u6 = upsample_block(bottleneck, f4, SIZE * 4)
    u7 = upsample_block(u6, f3, SIZE * 2)
    u8 = upsample_block(u7, f2, SIZE)
    u9 = upsample_block(u8, f1, SIZE / 2)

    outputs = layers.Conv2D(2, 1, padding="same", activation="softmax")(u9)

    model = tf.keras.Model(inputs, outputs, name="U-Net")  # pylint: disable=no-member

    return model


if __name__ == '__main__':
    ds, info = tfds.load(
        'galaxy_zoo3d', split=['train[2:3420]', 'train[3666:6999]'], with_info=True
    )
    ds_train, ds_test = ds[0], ds[1]

    if TRAIN_WITH == 'all':
        BUFFER_SIZE, BATCH_SIZE = 1000, 64
        TRAIN_LENGTH, VAL_SIZE, TEST_SIZE = 22360, 4992, 2461
    elif TRAIN_WITH == 'only':
        BUFFER_SIZE, BATCH_SIZE = 300, 16
        if MASK == 'spiral_mask':
            ds_train = ds_train.filter(
                lambda x: tf.reduce_max(x['spiral_mask']) >= MIN_VOTE
            )
            ds_test = ds_test.filter(
                lambda x: tf.reduce_max(x['spiral_mask']) >= MIN_VOTE
            )
            TRAIN_LENGTH, VAL_SIZE, TEST_SIZE = 4883, 1088, 551
        elif MASK == 'bar_mask':
            ds_train = ds_train.filter(
                lambda x: tf.reduce_max(x['bar_mask']) >= MIN_VOTE
            )
            ds_test = ds_test.filter(lambda x: tf.reduce_max(x['bar_mask']) >= MIN_VOTE)
            TRAIN_LENGTH, VAL_SIZE, TEST_SIZE = 3783, 832, 421

    train_dataset = ds_train.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = ds_test.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = (
        train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    )
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_batches = test_dataset.take(VAL_SIZE).batch(BATCH_SIZE)
    test_batches = test_dataset.skip(VAL_SIZE).take(TEST_SIZE).batch(BATCH_SIZE)

    unet_model = build_unet_model()

    unet_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics="accuracy",
    )

    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    VAL_SUBSPLITS = 5
    TEST_LENGTH = VAL_SIZE + TEST_SIZE
    VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE // VAL_SUBSPLITS

    model_history = unet_model.fit(
        train_batches,
        epochs=NUM_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        validation_data=validation_batches,
        # callbacks=[csv_log, early_stop, mcp_save_best, mcp_save_last]
    )
