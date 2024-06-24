import tensorflow as tf
import tensorflow_datasets as tfds

def get_dataloaders(cfg):
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(cfg.data.buffer_size)
    ds_train = ds_train.batch(cfg.training.batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(cfg.training.batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_test
