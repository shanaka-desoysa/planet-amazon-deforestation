import numpy as np
import os
import time # Timing

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend
from keras import applications


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class AmazonKerasClassifier:
    def __init__(self):
        self.losses = []
        self.classifier = Sequential()

    def save_vgg16_bottleneck_features(self, x_train, save_path, batch_size=128, model_summary=False):
        # build the VGG16 network and load imagenet weights
        # First time it will take longer, as it downloads the weights.
        if os.path.exists(save_path):
            print("{0} already exists. Bottleneck features not saved.".format(save_path))
            return

        model = applications.VGG16(include_top=False, weights='imagenet')
        if model_summary:
            model.summary()

        start_time = time.time()

        bottleneck_features_train = model.predict(x_train, batch_size=batch_size, verbose=1)

        np.save(open(save_path, 'wb'), bottleneck_features_train)

        print('save_bottleneck_features(): Time elapsed: {} seconds'.format(time.time()-start_time))

    def add_vgg16_model(self, img_size=(32, 32), img_channels=3):
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, img_channels))
        self.classifier.add(base_model)

    def add_conv_layer(self, img_size=(32, 32), img_channels=3):
        self.classifier.add(BatchNormalization(input_shape=(*img_size, img_channels)))

        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(128, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(256, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))


    # def add_flatten_layer(self):
    #     self.classifier.add(Flatten())

    def add_flatten_layer(self, input_shape):
        self.classifier.add(Flatten(input_shape=input_shape))

    def add_ann_layer(self, output_size):
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def train_model(self, x_train, y_train, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
        history = LossHistory()

        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=validation_split_size)

        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.classifier.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            validation_data=(X_valid, y_valid),
                            callbacks=[history, *train_callbacks, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def train_vgg16_full_model(self, x_train, y_train, img_size=(32, 32), img_channels=3, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
        history = LossHistory()
        # build the VGG16 network
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, img_channels))
        print('VGG16 Model loaded.')

        # add the model on top of the convolutional base
        self.classifier = Model(inputs= base_model.input, outputs=self.classifier(base_model.output))

        # set the first 15 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        for layer in self.classifier.layers[:15]:
            # print(layer.name)
            layer.trainable = False

        self.classifier.summary()

        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=validation_split_size)

        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.classifier.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            validation_data=(X_valid, y_valid),
                            callbacks=[history, *train_callbacks, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_weights(self, weight_file_path):
        self.classifier.save_weights(weight_file_path)

    def load_weights(self, weight_file_path):
        self.classifier.load_weights(weight_file_path)

    def predict(self, x_test):
        predictions = self.classifier.predict(x_test)
        return predictions

    def map_predictions(self, predictions, labels_map, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param labels_map: the map
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

    def close(self):
        backend.clear_session()
