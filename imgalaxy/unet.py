# pylint: disable=no-member
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
from wandb.keras import WandbMetricsLogger

from imgalaxy.cfg import MODELS_DIR
from imgalaxy.constants import BUFFER_SIZE, MASK, NUM_EPOCHS, THRESHOLD
from imgalaxy.helpers import dice, jaccard


class UNet:
    def __init__(
        self,
        loss: str = "sparse_categorical_crossentropy",
        dropout_rate: float = 0.3,
        num_epochs: int = NUM_EPOCHS,
        learning_rate: float = 0.0011,
        batch_size: int = 32,
        batch_normalization: bool = False,
        kernel_regularization: str = None,
        bias_regularization: str = None,
        activity_regularization: str = None,
        image_size: int = 128,
        n_filters: int = 128,
        mask: str = MASK,
        min_vote: int = 3,
    ):
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batch_normalization = batch_normalization
        self.image_size = image_size
        self.min_vote = min_vote
        self.n_filters = n_filters
        self.mask = mask
        self.kernel_regularization = kernel_regularization
        self.bias_regularization = bias_regularization
        self.activity_regularization = activity_regularization
        self.unet_model = self.build_unet_model()
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode="horizontal and vertical", seed=101),
            tf.keras.layers.RandomRotation(factor=(0, 1), seed=101),
            #tf.keras.layers.RandomZoom(
            #    height_factor=-0.23, width_factor=-0.23, seed=101
            #),
            # tf.keras.layers.RandomCrop(420, 420, seed=101)
        ])
        self.resize = tf.keras.Sequential([
            layers.Resizing(self.image_size, self.image_size),
        ])

        if self.mask == 'spiral_mask':
            self.TRAIN_LENGTH, self.VAL_SIZE, self.TEST_SIZE = 4883, 1088, 551
        elif self.mask == 'bar_mask':
            self.TRAIN_LENGTH, self.VAL_SIZE, self.TEST_SIZE = 3783, 832, 421

    def augment(self, image, mask):
        images_mask = tf.keras.layers.Concatenate(axis=2)([image, mask])
        images_mask = self.augmentation(images_mask)  
        
        image = images_mask[:, :, 0:3]
        mask = images_mask[:, :, 3:]
        
        mask = tf.cast(mask, 'uint8')
        
        return image, mask

    def binary_mask(self, mask, threshold: int = THRESHOLD):
        return tf.where(mask < threshold, tf.zeros_like(mask), tf.ones_like(mask))
    
    def load_image(self, datapoint, training=False):
        image = datapoint['image']
        mask = datapoint[self.mask]
        if training:
            image, mask = self.augment(image, mask)

        image = self.resize(image)
        mask = self.resize(mask)

        image = tf.cast(image, tf.float32) / 255.0
        mask = self.binary_mask(mask, THRESHOLD)

        return image, mask

    def double_conv_block(self, x, n_filters):
        x = layers.Conv2D(
            n_filters,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            # kernel_regularizer=self.kernel_regularization,
            # bias_regularizer=self.bias_regularization,
            # activity_regularizer=self.activity_regularization,
        )(x)
        if self.batch_normalization:
            x = layers.BatchNormalization()(x)

        x = layers.Conv2D(
            n_filters,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            # kernel_regularizer=self.kernel_regularization,
            # bias_regularizer=self.bias_regularization,
            # activity_regularizer=self.activity_regularization,
        )(x)
        if self.batch_normalization:
            x = layers.BatchNormalization()(x)
        return x

    def downsample_block(self, x, n_filters):
        f = self.double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(self.dropout_rate)(p)

        return f, p

    def upsample_block(self, x, conv_features, n_filters):
        x = layers.Conv2DTranspose(
            n_filters,
            3,
            2,
            padding="same",
            # kernel_regularizer=self.kernel_regularization,
            # bias_regularizer=self.bias_regularization,
            # activity_regularizer=self.activity_regularization,
        )(x)
        x = layers.concatenate([x, conv_features])
        x = layers.Dropout(self.dropout_rate)(x)
        x = self.double_conv_block(x, n_filters)

        return x

    def build_unet_model(self):
        inputs = layers.Input(shape=(self.image_size, self.image_size, 3))
        #inputs = self._augment(seed=123)(inputs, training=True)

        f1, p1 = self.downsample_block(inputs, self.n_filters // 2)
        f2, p2 = self.downsample_block(p1, self.n_filters)
        f3, p3 = self.downsample_block(p2, self.n_filters * 2)
        f4, p4 = self.downsample_block(p3, self.n_filters * 4)

        bottleneck = self.double_conv_block(p4, self.n_filters * 8)

        u6 = self.upsample_block(bottleneck, f4, self.n_filters * 4)
        u7 = self.upsample_block(u6, f3, self.n_filters * 2)
        u8 = self.upsample_block(u7, f2, self.n_filters)
        u9 = self.upsample_block(u8, f1, self.n_filters // 2)

        outputs = layers.Conv2D(2, 1, padding="same", activation="softmax")(u9)

        model = tf.keras.Model(inputs, outputs, name="U-Net")

        return model

    def train_pipeline(self):
        ds_train, ds_val, ds_test = tfds.load(
            'galaxy_zoo3d', split=['train[:75%]', 'train[75%:90%]', 'train[90%:]']
        )
        ds_train = ds_train.filter(
            lambda x: tf.reduce_max(x[self.mask]) >= self.min_vote
        )
        ds_val = ds_val.filter(lambda x: tf.reduce_max(x[self.mask]) >= self.min_vote)
        ds_test = ds_test.filter(lambda x: tf.reduce_max(x[self.mask]) >= self.min_vote)

        train_dataset = ds_train.map(
            lambda x: self.load_image(x, training=True),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        test_dataset = ds_test.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = ds_val.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)

        train_batches = (
            train_dataset.shuffle(BUFFER_SIZE).batch(self.batch_size).repeat()
        )
        train_batches = train_batches.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        test_batches = (
            test_dataset.shuffle(BUFFER_SIZE).batch(self.batch_size).repeat() 
        )
        val_batches = (
            val_dataset.shuffle(BUFFER_SIZE).batch(self.batch_size).repeat() 
        )

        self.unet_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss=self.loss,
            metrics=["accuracy", jaccard, dice],
            jit_compile=True,
        )

        STEPS_PER_EPOCH = self.TRAIN_LENGTH // self.batch_size
        VAL_SUBSPLITS = 5
        TEST_LENGTH = self.VAL_SIZE + self.TEST_SIZE
        VALIDATION_STEPS = TEST_LENGTH // self.batch_size // VAL_SUBSPLITS

        model_history = self.unet_model.fit(
            train_batches,
            epochs=self.num_epochs,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            validation_data=val_batches,
            callbacks=[
                WandbMetricsLogger(),
                tf.keras.callbacks.ModelCheckpoint(
                    MODELS_DIR / f"{self.mask}.keras"
                ),
            ],
        )

        return model_history, test_batches, train_batches
