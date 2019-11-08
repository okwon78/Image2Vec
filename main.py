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
import matplotlib.image as mpimg

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
        self.embedding_file = './embeddings.json'
        self.inverted_index_file = "./inverted_index.json"
        self.inverted_index = {}
        self.image_embedding = {}
        self.inverted_index = None

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

    def create_all_vec(self):
        if not os.path.exists(self.image_data_path):
            raise Exception("Invalid Operation", f"{self.image_data_path} does not exists")

        base_model = load_model(self.trained_weights_path)
        base_model.summary()
        intermediate_layer_model = Model(inputs=base_model.inputs, outputs=base_model.layers[1].output)

        self.image_embedding = {}

        classes = [f for f in listdir(self.image_data_path) if not f.startswith('.')]
        for elem in tqdm(classes):
            path = os.path.join(self.image_data_path, elem)
            images = [f for f in listdir(path) if ".jpg" in f]
            # print(images)
            count = 0

            for image in tqdm(images):
                image_path = os.path.join(path, image)
                self.image_embedding[image_path] = self.__get_vec(image_path, intermediate_layer_model).tolist()

                if count > 0:
                    break
                else:
                    count += 1

        with open(self.embedding_file, 'w') as fp:
            json.dump(self.image_embedding, fp)

        self.calac_knn(top_k=100)

    def calac_knn(self, top_k):
        if not os.path.exists(self.embedding_file):
            raise Exception("Invalid Operation", f"{self.embedding_file} does not exists")

        with open(self.embedding_file, 'r') as fp:
            self.image_embedding = json.load(fp)

        similarities = {}
        self.inverted_index = {}
        
        count = 0

        for key in tqdm(self.image_embedding.keys()):
            embedding = self.image_embedding[key]
            for target in self.image_embedding.keys():
                if key == target:
                    continue

                target_embedding = self.image_embedding[target]
                embedding = np.array(embedding).reshape(1, 32)
                target_embedding = np.array(target_embedding).reshape(1, len(target_embedding))
                similarities[target] = cosine_similarity(embedding, target_embedding)[0][0]

            self.inverted_index[key] = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[0:top_k]

            if count > 200:
                break
            else:
                count += 1

        with open(self.inverted_index_file, 'w') as fp:
            json.dump(self.inverted_index, fp)

    @staticmethod
    def __get_vec(image_path, model):

        img = image.load_img(image_path, target_size=(IMAGE_RESIZE, IMAGE_RESIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        intermediate_output = model.predict(x)
        return intermediate_output[0]

    def get_knn(self, filename, top_k=10):

        if self.inverted_index is None:
            with open(self.inverted_index_file, 'r') as fp:
                self.inverted_index = json.load(fp)

        neighbors = self.inverted_index[filename][0:top_k]

        self.draw_images(filename, neighbors)

        # for neighbor in neighbors:
        #     print(neighbor)

    @staticmethod
    def draw_images(target, images):
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

    # train model
    # imageSimilarity.download_file()
    # imageSimilarity.train()

    # extract embeddings
    # imageSimilarity.create_all_vec()

    # create inverted index for searching
    # imageSimilarity.calac_knn(10)

    imageSimilarity.get_knn("./geological_similarity/marble/MGN0Z.jpg", 10)
    imageSimilarity.get_knn("./geological_similarity/marble/W90SQ.jpg", 10)
    imageSimilarity.get_knn("./geological_similarity/marble/2G6SC.jpg", 10)
    imageSimilarity.get_knn("./geological_similarity/marble/PSQ1K.jpg", 10)

    # imageSimilarity.get_knn("./geological_similarity/marble/CSJY1.jpg", 10)
    # imageSimilarity.get_knn("./geological_similarity/marble/D9R51.jpg", 10)
    # imageSimilarity.get_knn("./geological_similarity/marble/IDUCB.jpg", 10)



if __name__ == '__main__':
    main()
