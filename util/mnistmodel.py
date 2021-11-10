from .datamodel import DataModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses, regularizers
import numpy as np


class MNISTModel(DataModel):
    def __init__(self):
        if type(self) == MNISTModel:
            raise Exception("<MNISTModel> must be subclassed.")

        super().__init__()

        (train_images, train_labels), (
            test_images,
            test_labels,
        ) = datasets.mnist.load_data()
        train_images = self._process_images(train_images)
        train_labels = train_labels
        test_images = self._process_images(test_images)
        test_labels = test_labels
        self.set_train_data(train_images, train_labels)
        self.set_test_data(test_images, test_labels)

    def get_name(self):
        return self.get_plain_name() + "_mnist"


class AlexNet(MNISTModel):
    def __init__(self):
        super().__init__()

        train_images, train_labels = self.get_train_data()

        model = models.Sequential()
        model.add(
            layers.experimental.preprocessing.Resizing(
                224,
                224,
                interpolation="bilinear",
                input_shape=train_images.shape[1:],
            )
        )
        model.add(layers.Conv2D(96, 11, strides=4, padding="same"))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(256, 5, strides=4, padding="same"))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(384, 3, strides=4, padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(384, 3, strides=4, padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(256, 3, strides=4, padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation="softmax"))

        model.compile(
            optimizer="adam",
            loss=losses.sparse_categorical_crossentropy,
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("alexnet")

    def train(self):
        train_images, train_labels = self.get_train_data()
        x_val = train_images[-2000:, :, :, :]
        y_val = train_labels[-2000:]

        self.model.fit(
            train_images,
            train_labels,
            batch_size=64,
            epochs=3,
            validation_data=(x_val, y_val),
        )
        test_loss, test_acc = self.test()
        print("Accuracy before faults:", test_acc)
        self.checkpoint_save()
        return test_loss, test_acc

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=64, epochs=3, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = tf.pad(images, [[0, 0], [2, 2], [2, 2]]) / 255
        images = tf.expand_dims(images, axis=3, name=None)
        images = tf.repeat(images, 3, axis=3)
        return images


class CNN(MNISTModel):
    def __init__(self):
        super().__init__()

        model = models.Sequential()
        model.add(
            layers.Conv2D(
                32, (5, 5), activation="relu", input_shape=(
                    28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation="softmax"))

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("cnn")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=100, epochs=5, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images.reshape((-1, 28, 28, 1))
        images = images / 255.0
        return images


class LeNet(MNISTModel):
    def __init__(self):
        super().__init__()

        model = models.Sequential()
        model.add(
            layers.Conv2D(
                filters=6,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(32, 32, 1),
            )
        )
        model.add(layers.AveragePooling2D())
        model.add(
            layers.Conv2D(
                filters=16,
                kernel_size=(
                    3,
                    3),
                activation="relu"))
        model.add(layers.AveragePooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(units=120, activation="relu"))
        model.add(layers.Dense(units=84, activation="relu"))
        model.add(layers.Dense(units=10, activation="softmax"))

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("lenet")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=100, epochs=5, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images[:, :, :, np.newaxis]
        images = images / 255.0
        images = np.pad(images, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant")
        return images


class NN(MNISTModel):
    def __init__(self):
        super().__init__()

        model = models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("nn")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=100, epochs=5, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class ResNet50(MNISTModel):
    def __init__(self):
        super().__init__()

        model = tf.keras.applications.ResNet50(
            include_top=True, input_tensor=None, weights=None,
            input_shape=(32,32,1), pooling=None, classes=10
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("resnet50")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=150, epochs=3, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = tf.pad(images, [[0, 0], [2, 2], [2, 2]]) / 255
        images = tf.expand_dims(images, axis=3, name=None)
        return images


class RNN(MNISTModel):
    def __init__(self):
        super().__init__()

        # Input sequences to RNN are the sequence of rows of MNIST digits
        # (treating each row of pixels as a timestep), and predict the digit's label.
        model = models.Sequential()
        model.add(layers.RNN(layers.LSTMCell(64), input_shape=(None, 28)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(10))

        model.compile(
            optimizer="sgd",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("rnn")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=512, epochs=5, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class VGG16(MNISTModel):
    def __init__(self):
        super().__init__()

        self.set_use_ds()

        model = tf.keras.models.Sequential()
        weight_decay = 0.0005
        num_classes = 10

        model.add(layers.Conv2D(64, (3, 3), padding='same',
                         input_shape=(28, 28, 1),kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax'))

        model.compile(
            optimizer="sgd",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("vgg16")

    def _train_model_ds(
            self,
            train_ds):
        self.model.fit(train_ds, epochs=5, verbose=self.get_verbose())
        return self.test()

    def _proc_ds(self, ds):
        return (ds
                .batch(batch_size=40, drop_remainder=False))

    def _process_images(self, images):
        images = images.reshape((-1, 28, 28, 1))
        images = images / 255.0
        return images

