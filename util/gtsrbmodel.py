from .datamodel import DataModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses, regularizers
import numpy as np
import pandas as pd
from skimage import io
import glob
import os


class GTSRBModel(DataModel):
    def __init__(self):
        if type(self) == GTSRBModel:
            raise Exception("<GTSRBModel> must be subclassed.")

        super().__init__()
        self.set_number_classes(43)

        root_dir = "GTSRB/"
        train_root_dir = root_dir + "Final_Training/Resized_Images/"
        test_root_dir = root_dir + "Final_Test/"

        train_images, train_labels = self.__read_train_data(train_root_dir)
        test_images, test_labels = self.__read_test_data(test_root_dir)

        train_images = self._process_images(train_images)
        test_images = self._process_images(test_images)
        self.set_train_data(train_images, train_labels)
        self.set_test_data(test_images, test_labels)

    def get_name(self):
        return self.get_plain_name() + "_gtsrb"

    def __read_train_data(self, train_root_dir):
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(train_root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            img = io.imread(img_path)
            label = self.__get_class(img_path)
            imgs.append(img)
            labels.append(label)

        train_images = np.array(imgs, dtype='float32')
        train_labels = np.array(labels)
        return train_images, train_labels

    def __read_test_data(self, test_root_dir):
        test = pd.read_csv(test_root_dir + "Labels/GT-final_test.csv", sep=';')

        # Load test dataset
        x_test = []
        y_test = []
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join(test_root_dir + "Resized_Images/", file_name)
            img = io.imread(img_path)

            x_test.append(img)
            y_test.append(class_id)

        test_images = np.array(x_test, dtype='float32')
        test_labels = np.array(y_test)
        return test_images, test_labels

    def __get_class(self, img_path):
            return int(img_path.split('/')[-2])


class AlexNet(GTSRBModel):
    def __init__(self):
        super().__init__()

        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
                2, 2), activation='relu', input_shape=(32, 32, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            keras.layers.Conv2D(filters=192, kernel_size=(
                3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(filters=384, kernel_size=(
                3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(filters=512, kernel_size=(
                3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.Conv2D(filters=512, kernel_size=(
                3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            keras.layers.Flatten(),

            keras.layers.Dropout(0.5),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dense(self.get_number_classes(), activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer="adam", metrics=['accuracy'])
        self.model = model
        self.set_name("alexnet")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=64, epochs=5, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class CNN(GTSRBModel):
    def __init__(self):
        super().__init__()

        model = models.Sequential()
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(512, (3, 3), activation='relu'))

        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(1024, (3, 3), padding='same', activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.get_number_classes()))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        self.model = model
        self.set_name("cnn")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=32, epochs=5, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class LeNet(GTSRBModel):
    def __init__(self):
        super().__init__()

        model = models.Sequential()
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.get_number_classes()))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        self.model = model
        self.set_name("lenet")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=64, epochs=5, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class NN(GTSRBModel):
    def __init__(self):
        super().__init__()

        model = models.Sequential([
            tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.get_number_classes(), activation='softmax'),
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        self.model = model
        self.set_name("nn")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=64, epochs=10, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class ResNet50(GTSRBModel):
    def __init__(self):
        super().__init__()

        model = tf.keras.applications.ResNet50(
            include_top=True, input_tensor=None, weights=None,
            input_shape=(32,32,3), pooling=None,
            classes=self.get_number_classes()
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
        self.model.fit(train_images, train_labels, batch_size=150, epochs=10, validation_split=0.1, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        return images


class RNN(GTSRBModel):
    def __init__(self):
        super().__init__()

        model = models.Sequential()
        model.add(layers.Conv1D(32, 4, activation='relu', padding='same', input_shape=(1024, 3)))
        model.add(layers.LSTM(32, return_sequences=True))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Conv1D(16, 8, activation="relu", padding='same'))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Conv1D(16, 8, activation="relu", padding='same'))
        model.add(layers.LSTM(128))
        model.add(layers.Dense(self.get_number_classes(), activation='sigmoid'))

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        self.model = model
        self.set_name("rnn")

    def _train_model(
            self,
            train_images,
            train_labels):
        self.model.fit(train_images, train_labels, batch_size=170, epochs=30, validation_split=0.1, verbose=self.get_verbose())
        return self.test()

    def _process_images(self, images):
        images = images / 255.0
        images = images.reshape(images.shape[0], 1024, 3)
        return images


class VGG16(GTSRBModel):
    def __init__(self):
        super().__init__()

        self.set_use_ds()

        model = tf.keras.models.Sequential()
        weight_decay = 0.0005

        model.add(layers.Conv2D(64, (3, 3), padding='same',
                         input_shape=(32,32,3),kernel_regularizer=regularizers.l2(weight_decay)))
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

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(self.get_number_classes()))
        model.add(layers.Activation('softmax'))

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.model = model
        self.set_name("vgg16")

    def _train_model_ds(
            self,
            train_ds):
        self.model.fit(train_ds, epochs=10, verbose=self.get_verbose())
        return self.test()

    def _proc_ds(self, ds):
        return (ds
                .batch(batch_size=128, drop_remainder=False))

