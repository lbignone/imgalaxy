"""Helper methods"""
import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from sklearn.metrics import confusion_matrix, jaccard_score

tf.config.run_functions_eagerly(True)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def evaluate_model(dataset, model, num=1):
    """Evaluate model after a run is completed."""
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


tp = keras.metrics.TruePositives()
fp = keras.metrics.FalsePositives()
fn = keras.metrics.FalseNegatives()


def jaccard(y_true, y_pred):
    """Jaccard index to compute after each epoch."""
    tp.update_state(y_true, y_pred[0])
    fp.update_state(y_true, y_pred[0])
    fn.update_state(y_true, y_pred[0])

    return tp.result() / (tp.result() + fp.result() + fn.result())


def dice(y_true, y_pred):
    """Dice coefficient to compute after each epoch."""
    _jaccard = jaccard(y_true, y_pred)
    return 2 * _jaccard / (1 + _jaccard)


class JaccardScoreCallback(keras.callbacks.Callback):
    """Computes the Jaccard score and logs the results to TensorBoard."""

    def __init__(self, test_batch):
        self._x_test = [b[0] for b in test_batch.as_numpy_iterator()]
        self._y_test = [b[1] for b in test_batch.as_numpy_iterator()]
        self.x_test = [image for image, _ in tfds.as_numpy(test_batch)]
        self.y_test = [label for _, label in tfds.as_numpy(test_batch)]
        self.keras_metric = keras.metrics.Mean("jaccard_score")
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        self.keras_metric.reset_state()
        predictions = self.model.predict(self.x_test)
        jaccard_value = jaccard_score(
            np.argmax(predictions, axis=-1), self.y_test, average=None
        )
        self.keras_metric.update_state(jaccard_value)
        wandb.log(
            {self.keras_metric.name: self.keras_metric.result().numpy().astype(float)}
        )
