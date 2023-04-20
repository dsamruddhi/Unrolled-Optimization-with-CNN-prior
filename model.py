import os
from datetime import datetime
import numpy as np
from abc import ABC
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from config import Config
from dataloader.data_loader import DataLoader
from utils.plot_utils import PlotUtils
from metric_functions.metrics import Metrics


class SimpleResidualCNN(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv1 = Conv2D(64, 3, padding="SAME", kernel_initializer=tf.initializers.GlorotNormal())
        self.b1 = BatchNormalization()
        self.act1 = Activation("relu")
        self.conv2 = Conv2D(64, 3, padding="SAME", kernel_initializer=tf.initializers.GlorotNormal())
        self.b2 = BatchNormalization()
        self.act2 = Activation("relu")

    def call(self, inputs, **kwargs):
        zt1 = inputs
        o1 = self.act1(self.b1(self.conv1(zt1)))
        xt1 = self.act2(self.b2(self.conv2(o1)))
        xt1 = xt1 + zt1
        return xt1


class UNetCNN(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv1 = Conv2D(64, 3, padding="SAME", kernel_initializer=tf.initializers.GlorotNormal())
        self.b1 = BatchNormalization()
        self.act1 = Activation("relu")
        self.p1 = MaxPooling2D(2)

        self.conv2 = Conv2D(64, 3, padding="SAME", kernel_initializer=tf.initializers.GlorotNormal())
        self.b2 = BatchNormalization()
        self.act2 = Activation("relu")
        self.p2 = MaxPooling2D(2)

        self.conv3 = Conv2D(64, 3, padding="SAME", kernel_initializer=tf.initializers.GlorotNormal())
        self.b3 = BatchNormalization()
        self.act3 = Activation("relu")
        self.u3 = UpSampling2D(2)

        self.conv4 = Conv2D(64, 3, padding="SAME", kernel_initializer=tf.initializers.GlorotNormal())
        self.b4 = BatchNormalization()
        self.act4 = Activation("relu")
        self.u4 = UpSampling2D(2)

        self.conv5 = Conv2D(64, 3, padding="SAME", kernel_initializer=tf.initializers.GlorotNormal())
        self.b5 = BatchNormalization()
        self.act5 = Activation("relu")

        self.conv6 = Conv2DTranspose(1, 3, padding="VALID")

    def call(self, inputs, **kwargs):
        zt1 = inputs
        o1 = self.p1(self.act1(self.b1(self.conv1(zt1))))  # 25x25
        o2 = self.p2(self.act2(self.b2(self.conv2(o1))))   # 12x12
        o3 = self.u3(self.act3(self.b3(self.conv3(o2))))   # 24x24
        o4 = self.u4(self.act4(self.b4(self.conv4(o3))))   # 48x48
        o5 = self.act5(self.b5(self.conv5(o4)))
        o6 = self.conv6(o5)                                # 50x50
        o6 = o6 + zt1
        return o6


class ProximalLayer(layers.Layer):
    """
    Single layer for an unrolled optimization network performing least squares minimization with l1 constraints
    solved using the Proximal Gradient Algorithm.
    """

    def __init__(self, A, eta, prior):

        super(ProximalLayer, self).__init__()
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.eta = tf.Variable(initial_value=eta, dtype=tf.float32, trainable=True)
        self.prior = prior

    def call(self, inputs, **kwargs):
        [xt, y] = inputs
        nabla_f = tf.matmul(tf.transpose(self.A), tf.matmul(self.A, xt) - y[0])
        zt1 = xt - (self.eta * nabla_f)
        zt1 = tf.reshape(zt1, (-1, 50, 50))
        zt1 = zt1[..., np.newaxis]
        xt1 = self.prior(zt1)
        xt1 = tf.reshape(xt1[:, :, :, 0], (-1, 2500, 1))
        return [xt1, y]


class UnrolledOptimization(ABC):

    def __init__(self):
        super().__init__()

        # Data and its attributes
        self.train_dataset = None
        self.test_dataset = None

        self.data_generator = ImageDataGenerator()

        # Model and its attributes
        self.model_path = Config.config["model"]["model_path"]
        self.experiment_name = Config.config["model"]["experiment_name"]
        self.model = None
        self.optimizer = None

        self.eta = 0.01
        self.num_iters = 8

        # Training
        self.batch_size = Config.config["train"]["batch_size"]

        # Logging
        self.file_writer = None

    def load_data(self, show_data=False):
        gen_data_train, gen_data_test,\
            real_data_train, real_data_test, \
            measurements_train, measurements_test = DataLoader().main(show_data)

        train_dataset = tf.data.Dataset.from_tensor_slices((gen_data_train, real_data_train, measurements_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((gen_data_test, real_data_test, measurements_test))

        self.train_dataset = train_dataset.batch(self.batch_size, drop_remainder=True)
        self.test_dataset = test_dataset.batch(self.batch_size, drop_remainder=True)

        self.A = DataLoader.rytov_model()

    def build(self):

        def _unrolled_proxgrad():
            X = tf.keras.layers.Input((2500, 1), dtype=tf.float32)
            Y = tf.keras.layers.Input((1560, 1), dtype=tf.float32)
            X_, Y_ = X, Y
            x_recs = []
            for iter in range(0, self.num_iters):
                prior = UNetCNN()
                X_, Y_ = ProximalLayer(self.A, self.eta, prior)([X_, Y_])
                x_recs.append(X_)
            model = tf.keras.Model(inputs=[X, Y], outputs=[X_, Y_])
            return model

        self.model = _unrolled_proxgrad()
        print(self.model.summary())

        tf.keras.utils.plot_model(self.model, "model.png", show_shapes=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    def log(self):
        log_dir = os.path.join(Config.config["model"]["model_path"], "logs")
        self.summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                         Config.config['model']['experiment_name'],
                                                                         datetime.now().strftime("%Y%m%d-%H%M%S")))

    def train(self, steps):

        train_ds = self.train_dataset.repeat(3000).as_numpy_iterator()

        for step in tf.range(steps):
            tf.print("Step: ", step)
            step = tf.cast(step, tf.int64)

            gen_batch, real_batch, measurement_batch = train_ds.next()
            gen_batch[gen_batch < 0] = 0
            gen_batch = gen_batch.reshape((gen_batch.shape[0], 2500, 1))
            real_batch = real_batch.reshape((real_batch.shape[0], 2500, 1))

            with tf.GradientTape() as tape:
                out_batch, _ = self.model([gen_batch, measurement_batch])
                loss = tf.reduce_mean(tf.square(out_batch - real_batch))

            network_gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(network_gradients, self.model.trainable_variables))

            tf.print("loss value: ", loss)

            with self.summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)

        # print("Trainable variables: ", self.model.trainable_variables)

    def evaluate(self, index):

        train_ds = self.test_dataset.repeat(1).as_numpy_iterator()

        gen_batch, real_batch, measurement_batch = train_ds.next()
        gen_batch[gen_batch < 0] = 0

        gen_batch = gen_batch.reshape((gen_batch.shape[0], 2500, 1))
        real_batch = real_batch.reshape((real_batch.shape[0], 2500, 1))

        out_batch, _ = self.model([gen_batch, measurement_batch])

        gt = real_batch[index].reshape((50, 50), order='F')
        start = gen_batch[index].reshape((50, 50), order='F')
        current = np.asarray(out_batch[index]).reshape((50, 50), order='F')

        current1 = np.copy(current)
        current1[current1 < 0] = 0

        psnr_start = Metrics.psnr(np.asarray(gt), np.asarray(start))
        psnr_current = Metrics.psnr(np.asarray(gt), np.asarray(current1))

        PlotUtils.plot_output(gt, start, current1, psnr_start, psnr_current)


if __name__ == '__main__':

    """ TF / GPU config """
    tf.random.set_seed(1234)
    tf.keras.backend.clear_session()
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = InteractiveSession(config=config)

    experiment = UnrolledOptimization()
    experiment.load_data(show_data=False)
    experiment.build()
    experiment.log()
    experiment.train(3000)
    index = 11
    experiment.evaluate(index)
