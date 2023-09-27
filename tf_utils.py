import tensorflow as tf
import numpy as np

def get_tf_dataset(inputs, batch_size, shuffle=True):

    x, y = inputs
    n = x.shape[0]

    data = tf.data.Dataset.from_tensor_slices(x.astype(np.float32))
    labels = tf.data.Dataset.from_tensor_slices(y.astype(np.float32))

    if shuffle:
        dataset = (tf.data.Dataset.zip((data, labels))
                    .shuffle(n).batch(batch_size).prefetch(4))
    else:
        dataset = (tf.data.Dataset.zip((data, labels)).batch(batch_size).prefetch(4))
    return dataset