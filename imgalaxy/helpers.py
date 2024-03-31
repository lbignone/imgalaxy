"""Helper methods"""
import keras
import numpy as np
import tensorflow as tf
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


def jaccard(y_true, y_pred):
    """Jaccard index to compute after each epoch."""
    tp = keras.metrics.TruePositives()
    fp = keras.metrics.FalsePositives()
    fn = keras.metrics.FalseNegatives()

    y_hats = tf.math.argmax(y_pred, axis=-1)
    tp.update_state(y_true, y_hats)
    fp.update_state(y_true, y_hats)
    fn.update_state(y_true, y_hats)

    score = tp.result() / (tp.result() + fp.result() + fn.result())

    return score.numpy()


def dice(y_true, y_pred):
    """Dice coefficient to compute after each epoch."""
    tp = keras.metrics.TruePositives()
    fp = keras.metrics.FalsePositives()
    fn = keras.metrics.FalseNegatives()

    y_hats = tf.math.argmax(y_pred, axis=-1)
    tp.update_state(y_true, y_hats)
    fp.update_state(y_true, y_hats)
    fn.update_state(y_true, y_hats)

    score = 2 * tp.result() / (2 * tp.result() + fp.result() + fn.result())

    return score.numpy()
