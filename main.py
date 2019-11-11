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
from tqdm import tqdm

import os
import requests
import json

from zipfile import ZipFile
from os import listdir

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from shutil import copyfile
import random

from annoy import AnnoyIndex

NUM_CLASSES = 6
IMAGE_RESIZE = 224

NUM_EPOCHS = 100
EMBEDDING_SIZE = 32
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
        self.data_dir = "./data"
        self.train_dir = "./data/train"
        self.validation_dir = "./data/valid"

        self.embedding_file = './embeddings.json'
        self.inverted_index_file = "./inverted_index.json"
        self.inverted_index = {}
        self.image_embedding = {}
        self.inverted_index = None
        self.annoyIndex = None
        self.annyIndexTofilenames = None
        self.intermediate_layer_model = None

        if not os.path.exists('./working'):
            os.mkdir('./working')

    def download_file(self):
        """
        Downloads geological_similarity.zip file and Extract zip file
        """
        if os.path.exists(self.image_data_path):
            return
        r = requests.get(data_url, allow_redirects=True)
        with open(self.zipfile_name, 'wb') as f:
            f.write(r.content)

        with ZipFile(self.zipfile_name, 'r') as f:
            f.extractall()

        os.remove(self.zipfile_name)

    def split_data(self):
        """
        splits geological_similarity into train and validation data
        """
        if not os.path.exists(self.image_data_path):
            raise Exception("Data do not exist")

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)
        if not os.path.exists(self.validation_dir):
            os.mkdir(self.validation_dir)

        classes = [f for f in listdir(self.image_data_path) if not f.startswith('.')]
        for elem in tqdm(classes):
            path = os.path.join(self.image_data_path, elem)
            images = [f for f in listdir(path) if ".jpg" in f]
            total_count = len(images)
            split_num = int(total_count / 3)

            train_class_dir = os.path.join(self.train_dir, elem)
            if not os.path.exists(train_class_dir):
                os.mkdir(train_class_dir)

            validation_class_dir = os.path.join(self.validation_dir, elem)
            if not os.path.exists(validation_class_dir):
                os.mkdir(validation_class_dir)

            random.shuffle(images)

            train_data = images[0:split_num]
            train_data = [(os.path.join(self.image_data_path, elem, f), os.path.join(train_class_dir, f)) for f in
                          train_data]

            for pair in train_data:
                copyfile(pair[0], pair[1])

            validation_data = images[split_num:-1]
            validation_data = [(os.path.join(self.image_data_path, elem, f), os.path.join(validation_class_dir, f)) for
                               f in validation_data]

            for pair in validation_data:
                copyfile(pair[0], pair[1])

    def build_model(self):
        """
        builds train model
        resnet + embedding layer + softmax layer
        """
        model = Sequential()
        resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet')
        # resnet.summary()

        model.add(resnet)
        model.add(Dense(EMBEDDING_SIZE, activation='relu', name="embedding"))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        model.layers[0].trainable = False
        model.summary()

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_generator(self):
        """
        Creates train_generator, validation_generator
        """
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_generator = data_generator.flow_from_directory(
            self.train_dir,
            target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
            shuffle=True,
            batch_size=100,
            class_mode='categorical')

        self.validation_generator = data_generator.flow_from_directory(
            self.validation_dir,
            target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
            shuffle=True,
            batch_size=100,
            class_mode='categorical')

    def train(self):
        """
        Transfer learning
        """
        if os.path.exists(self.trained_weights_path):
            model = load_model(self.trained_weights_path)
        else:
            model = self.build_model()

        self.create_generator()

        cb_checkpointer = ModelCheckpoint(filepath=self.trained_weights_path, monitor='val_loss', save_best_only=True,
                                          mode='auto')
        fit_history = model.fit_generator(
            self.train_generator,
            steps_per_epoch=100,
            epochs=NUM_EPOCHS,
            validation_data=self.validation_generator,
            validation_steps=100,
            callbacks=[cb_checkpointer]
        )

        model.load_weights(self.trained_weights_path)

    def create_all_vec(self):
        """
        feed-forward to extract all the embeddings from all images in geological_similarity directory
        """
        if not os.path.exists(self.image_data_path):
            raise Exception("Invalid Operation", f"{self.image_data_path} does not exists")

        self.image_embedding = {}

        classes = [f for f in listdir(self.image_data_path) if not f.startswith('.')]
        for elem in tqdm(classes):
            path = os.path.join(self.image_data_path, elem)
            images = [f for f in listdir(path) if ".jpg" in f]

            for image in tqdm(images):
                image_path = os.path.join(path, image)
                self.image_embedding[image_path] = self.__get_vec(image_path)

        with open(self.embedding_file, 'w') as fp:
            json.dump(self.image_embedding, fp)

    def calac_knn(self, top_k):
        """
        1. gets cosine distances
        2. saves top k nearest neighbors using cosine_similarity of sklearn into  inverted_index
        """
        if not os.path.exists(self.embedding_file):
            raise Exception("Invalid Operation", f"{self.embedding_file} does not exists")

        with open(self.embedding_file, 'r') as fp:
            self.image_embedding = json.load(fp)

        similarities = {}
        self.inverted_index = {}

        for key in tqdm(self.image_embedding.keys()):

            embedding = self.image_embedding[key]
            for target in self.image_embedding.keys():
                if key == target:
                    continue

                target_embedding = self.image_embedding[target]
                embedding = np.array(embedding).reshape(1, EMBEDDING_SIZE)
                target_embedding = np.array(target_embedding).reshape(1, len(target_embedding))
                similarities[target] = cosine_similarity(embedding, target_embedding)[0][0]

            self.inverted_index[key] = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[0:top_k]

        with open(self.inverted_index_file, 'w') as fp:
            json.dump(self.inverted_index, fp)

    def calc_knn_annoy(self, top_k):
        """
        1. build annoy tree to compute k nearest neighbors
        2. saves top k nearest neighbors using annoy into inverted_index
        """
        if not os.path.exists(self.embedding_file):
            raise Exception("Invalid Operation", f"{self.embedding_file} does not exists")

        with open(self.embedding_file, 'r') as fp:
            self.image_embedding = json.load(fp)

        self.annoyIndex = AnnoyIndex(EMBEDDING_SIZE, 'angular')

        self.annyIndexTofilenames = {}
        for idx, key in tqdm(enumerate(self.image_embedding.keys())):
            self.annoyIndex.add_item(idx, self.image_embedding[key])
            self.annyIndexTofilenames[idx] = [key, 0]

        self.annoyIndex.build(-1)

        self.inverted_index = {}
        print(len(self.image_embedding))
        for idx, key in tqdm(enumerate(self.image_embedding.keys())):
            indexes = self.annoyIndex.get_nns_by_item(idx, top_k + 1)

            values = []
            for target in indexes:
                if idx == target:
                    continue
                values.append(self.annyIndexTofilenames[target])

            self.inverted_index[key] = values

        with open(self.inverted_index_file, 'w') as fp:
            json.dump(self.inverted_index, fp)

    def __get_vec(self, image_path):
        """
        returns a embedding, given a image
        """
        if self.intermediate_layer_model is None:
            base_model = load_model(self.trained_weights_path)
            self.intermediate_layer_model = Model(inputs=base_model.inputs, outputs=base_model.layers[1].output)

        img = image.load_img(image_path, target_size=(IMAGE_RESIZE, IMAGE_RESIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        intermediate_output = self.intermediate_layer_model.predict(x)
        return intermediate_output[0].tolist()

    def get_knn(self, filename, top_k=10):
        """
        Checks nearest neighbors from pre-built inverted index file
        """
        if self.inverted_index is None:
            with open(self.inverted_index_file, 'r') as fp:
                self.inverted_index = json.load(fp)

        neighbors = self.inverted_index[filename][0:top_k]

        self.draw_images(neighbors)

        for neighbor in neighbors:
            print(neighbor)

    def get_similar_image(self, new_image_path, top_k):
        """
        1. gets a embedding vector, given a image
        2. returns similar images using annoy
        """
        if self.annoyIndex is None:
            self.calc_knn_annoy(top_k)

        embeddings = self.__get_vec(new_image_path)
        indexes = self.annoyIndex.get_nns_by_vector(embeddings, top_k)
        nn = []
        for idx in indexes:
            nn.append(self.annyIndexTofilenames[idx])

        self.draw_images(nn)

    @staticmethod
    def draw_images(images):
        """
        draws images using matplotlib
        """
        if images is None:
            raise Exception("invalid input")

        fig = plt.figure(figsize=(10, 10))
        count = len(images)
        columns = int(np.sqrt(count))
        if count == columns * columns:
            rows = columns
        else:
            rows = columns + 1

        for idx, pos in enumerate(range(1, columns * rows + 1)):
            if idx == len(images):
                break

            img = plt.imread(images[idx][0])
            fig.add_subplot(rows, columns, pos)
            plt.imshow(img)
        plt.show(block=True)


def main():
    imageSimilarity = ImageSimilarity()

    # download train data
    imageSimilarity.download_file()

    # split data into train and validation
    imageSimilarity.split_data()

    # train model
    imageSimilarity.train()

    # extract embeddings
    imageSimilarity.create_all_vec()

    # create inverted index with cosine similarities for searching
    imageSimilarity.calac_knn(100)

    # create inverted index with ANN for searching
    imageSimilarity.calc_knn_annoy(100)

    # check top 10 neighbors
    imageSimilarity.get_knn("./geological_similarity/marble/PDF9R.jpg", 10)
    imageSimilarity.get_knn("./geological_similarity/gneiss/1OK58.jpg", 10)
    imageSimilarity.get_knn("./geological_similarity/andesite/0JDL9.jpg", 10)

    # get similar images from a unseen image
    imageSimilarity.get_similar_image("./0HBTF.jpg", 10)


if __name__ == '__main__':
    main()
