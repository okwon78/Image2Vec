from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
import os

NUM_CLASSES = 6
IMAGE_RESIZE = 224

NUM_EPOCHS = 100

LOSS_METRICS = ['accuracy']


# http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip

class ImageSimilarity:
    def __init__(self):
        self.intermediate_layer_model = None
        self.train_generator = None
        self.validation_generator = None
        self.trained_weights_path = './working/best.hdf5'
        self.image_data_path = './geological_similarity'

    def download_file(self):
        pass

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

    def data_prepare(self):
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
        model = self.build_model()

        if os.path.exists(self.trained_weights_path):
            model.load_weights(self.trained_weights_path)

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

    def get_vec(self, image_path):
        model = self.build_model()

        if os.path.exists(self.trained_weights_path):
            model.load_weights(self.trained_weights_path)

        img = image.load_img(image_path, target_size=(IMAGE_RESIZE, IMAGE_RESIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        print(x.shape)
        self.intermediate_layer_model = Model(inputs=model.inputs,
                                              outputs=model.layers[1].output)
        intermediate_output = self.intermediate_layer_model.predict(x)

        print(intermediate_output[0])


def main():
    imageSimilarity = ImageSimilarity()

    # imageSimilarity.data_prepare()
    # imageSimilarity.train()

    imageSimilarity.get_vec('./geological_similarity/andesite/0FVDN.jpg')


if __name__ == '__main__':
    main()
