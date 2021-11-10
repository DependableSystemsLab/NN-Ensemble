from .datamodel import DataModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses, regularizers, optimizers
import numpy as np
from .ResNet18Model import ResNet18Model


class CIFAR10Model(DataModel):
    def __init__(self):
        if type(self) == CIFAR10Model:
            raise Exception("<CIFAR10Model> must be subclassed.")

        super().__init__()

        (train_images, train_labels), (
            test_images,
            test_labels,
        ) = datasets.cifar10.load_data()
        train_images = self._process_images(train_images)
        train_labels = train_labels
        test_images = self._process_images(test_images)
        test_labels = test_labels
        self.set_train_data(train_images, train_labels)
        self.set_test_data(test_images, test_labels)

    def get_name(self):
        return self.get_plain_name() + "_cifar10"


class ConvNet(CIFAR10Model):
    def __init__(self):
        super().__init__()

        model = models.Sequential()
        model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        self.model = model
        self.set_name("convnet")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=128, epochs=150, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class DeconvNet(CIFAR10Model):
    def __init__(self):
        super().__init__()

        model = models.Sequential()
        model.add(layers.Conv2D(input_shape=(32,32,3), filters=96, kernel_size=(3,3)))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=96, kernel_size=(3,3), strides=2))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(filters=192, kernel_size=(3,3)))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=192, kernel_size=(3,3), strides=2))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(256))
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(10, activation="softmax"))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        self.model = model
        self.set_name("deconvnet")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=128, epochs=150, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class MobileNet(CIFAR10Model):
    def __init__(self):
        super().__init__()

        mobile = tf.keras.applications.mobilenet.MobileNet(include_top=True,
                                                           input_shape=(32,32,3),
                                                           pooling='max', weights=None,#'imagenet',
                                                           alpha=1, depth_multiplier=1,dropout=.2)
        x=mobile.layers[-1].output
        x=keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=0.001 )(x)
        predictions=layers.Dense (10, activation='softmax')(x)
        model = keras.Model(inputs=mobile.input, outputs=predictions)
        for layer in model.layers:
            layer.trainable=True

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("mobilenet")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=150, epochs=100, validation_split=0.1, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class ResNet18(CIFAR10Model):
    def __init__(self):
        super().__init__()

        model = ResNet18Model(10)
        model.build(input_shape = (None,32,32,3))

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("resnet18")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=256, epochs=50, validation_split=0.1, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class ResNet50(CIFAR10Model):
    def __init__(self):
        super().__init__()

        model = tf.keras.applications.ResNet50(
            include_top=True, input_tensor=None, weights=None,
            input_shape=(32,32,3), pooling=None, classes=10
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
        self.model.fit(train_images, train_labels, batch_size=150, epochs=100, validation_split=0.1, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class VGG3(CIFAR10Model):
    def __init__(self):
        super().__init__()

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(10, activation='softmax'))

        opt = optimizers.SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        self.model = model
        self.set_name("vgg3")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=128, epochs=50, validation_split=0.1, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class VGG11(CIFAR10Model):
    def __init__(self):
        super().__init__()

        self.set_use_ds()

        model = tf.keras.models.Sequential()
        weight_decay = 0.0005
        num_classes = 10

        model.add(layers.Conv2D(64, (3, 3), padding='same',
                         input_shape=(32,32,3),kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax'))

        model.compile(
            optimizer="sgd",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("vgg11")

    def _train_model_ds(
            self,
            train_ds):
        self.model.fit(train_ds, epochs=15, verbose=2)#self.get_verbose())
        return self.test()

    def _proc_ds(self, ds):
        return (ds
                .batch(batch_size=40, drop_remainder=False))


class VGG16(CIFAR10Model):
    def __init__(self):
        super().__init__()

        self.set_use_ds()

        model = tf.keras.models.Sequential()
        weight_decay = 0.0005
        num_classes = 10

        model.add(layers.Conv2D(64, (3, 3), padding='same',
                         input_shape=(32,32,3),kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax'))

        model.compile(
            optimizer="sgd",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("vgg16")

    def _train_model_ds(
            self,
            train_ds):
        self.model.fit(train_ds, epochs=15, verbose=2)#self.get_verbose())
        return self.test()

    def _proc_ds(self, ds):
        return (ds
                .batch(batch_size=40, drop_remainder=False))

class DenseNet(CIFAR10Model):
    def __init__(self):
        super().__init__()

        model = tf.keras.applications.DenseNet121(
            include_top=True, input_tensor=None, weights=None,
            input_shape=(32,32,3), pooling=None, classes=10
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("densenet")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=128, epochs=20, validation_split=0.1, verbose=2)
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images

