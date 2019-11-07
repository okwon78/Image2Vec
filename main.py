from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf

import os
import requests

from zipfile import ZipFile
from os import listdir

from scipy import spatial

NUM_CLASSES = 6
IMAGE_RESIZE = 224

NUM_EPOCHS = 100

LOSS_METRICS = ['accuracy']

data_url = "http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip"


class ImageSimilarity:
    def __init__(self):
        self.intermediate_layer_model = None
        self.train_generator = None
        self.validation_generator = None
        self.trained_weights_path = './working/best.hdf5'
        self.image_data_path = './geological_similarity'
        self.zipfile_name = "./geological_similarity.zip"
        self.inverted_index = {}

        if not os.path.exists('./working'):
            os.mkdir('./working')

    def download_file(self):
        if os.path.exists(self.image_data_path):
            return
        r = requests.get(data_url, allow_redirects=True)
        with open(self.zipfile_name, 'wb') as f:
            f.write(r.content)

        with ZipFile(self.zipfile_name, 'r') as f:
            f.extractall()

        os.remove(self.zipfile_name)

    def build_model(self):
        model = Sequential()
        resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet')
        # resnet.summary()

        model.add(resnet)
        model.add(Dense(32, activation='relu', name="embedding"))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        model.layers[0].trainable = False
        model.summary()

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def __data_prepare(self):
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_generator = data_generator.flow_from_directory(
            self.image_data_path,
            target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
            shuffle=True,
            batch_size=100,
            class_mode='categorical')

        self.validation_generator = data_generator.flow_from_directory(
            self.image_data_path,
            target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
            shuffle=True,
            batch_size=1,
            class_mode='categorical')

    def train(self):

        if os.path.exists(self.trained_weights_path):
            model = load_model(self.trained_weights_path)
        else:
            model = self.build_model()

        self.__data_prepare()

        cb_checkpointer = ModelCheckpoint(filepath=self.trained_weights_path, monitor='val_loss', save_best_only=True,
                                          mode='auto')

        fit_history = model.fit_generator(
            self.train_generator,
            steps_per_epoch=30,
            epochs=NUM_EPOCHS,
            validation_data=self.validation_generator,
            validation_steps=3000,
            callbacks=[cb_checkpointer]
        )

        model.load_weights(self.trained_weights_path)

    def get_all_vec(self):
        if not os.path.exists(self.image_data_path):
            raise Exception("Invalid Operation", f"{self.image_data_path} does not exists")

        model = load_model(self.trained_weights_path)

        model.summary()
        # if os.path.exists(self.trained_weights_path):
        #     model.load_weights(self.trained_weights_path, by_name=True)

        intermediate_layer_model = Model(inputs=model.inputs, outputs=model.layers[1].output)

        self.inverted_index = {}

        classes = [f for f in listdir(self.image_data_path) if not f.startswith('.')]
        for elem in classes:
            path = os.path.join(self.image_data_path, elem)
            images = [f for f in listdir(path) if ".jpg" in f]
            # print(images)
            for image in images:
                image_path = os.path.join(path, image)
                self.inverted_index[image_path] = self.get_vec(image_path, intermediate_layer_model)
                print(image_path, " ", self.inverted_index[image_path])

    def get_vec(self, image_path, model):

        img = image.load_img(image_path, target_size=(IMAGE_RESIZE, IMAGE_RESIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        intermediate_output = model.predict(x)
        return intermediate_output[0]


def main():
    imageSimilarity = ImageSimilarity()

    imageSimilarity.download_file()
    imageSimilarity.train()

    imageSimilarity.get_all_vec()

    # imageSimilarity.get_vec('./geological_similarity/andesite/0FVDN.jpg')


if __name__ == '__main__':
    main()
