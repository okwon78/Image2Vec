# Geological Image Similarity

I tried to identify similar images for a given image using CNN.
Especially, I used pre-trained resent50 becasue of lack of time resources.
The reason why I chose resent50 is that It doesn't hurt performance much, becuse of skip connections, even though resnet50 is one of the heavy models.

On top of the trained resnet with imagenet which does not include original top, I stacked two more layers. One is for feature extraction named embedding. 
The next is softmax layer for training new data. And I did not train resnet layers. I only train added two layers. Thus I set trainable variable of resnet model to False.

```python
model = Sequential()
resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet')

model.add(resnet)
model.add(Dense(32, activation='relu', name="embedding"))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.layers[0].trainable = False
```

During training, I used all data. In general, train, validation and test dataset are required to estimate model performance or to compare other models   
However, I don't care about measuring model performance. So I do not split dataset into train, validation and test.


```python
cb_checkpointer = ModelCheckpoint(filepath=self.trained_weights_path, monitor='val_loss', save_best_only=True, mode='auto')

fit_history = model.fit_generator(
    self.train_generator,
    steps_per_epoch=30,
    epochs=NUM_EPOCHS,
    validation_data=self.validation_generator,
    validation_steps=3000,
    callbacks=[cb_checkpointer]
)
```

After training, I builded new model from previous model to extract image embeddings

```python
base_model = load_model(self.trained_weights_path)
intermediate_layer_model = Model(inputs=base_model.inputs, outputs=base_model.layers[1].output)
```
And then, I extracted embeddings from all of images

```python
img = image.load_img(image_path, target_size=(IMAGE_RESIZE, IMAGE_RESIZE))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

intermediate_output = model.predict(x)
```

Finally, I builded a inverted index; Key is file path and value is a sorted list by cosine similarities.
The reason why I chose cosine distance instand of euclidean distance is that In high dimension space, cosine distance is a better choice because of curse of dimension.


```python
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
```
## To run code

```bash
$ python main.py
```

## main method
```python
def main():
    imageSimilarity = ImageSimilarity()

    # train model
    imageSimilarity.download_file()
    imageSimilarity.train()

    # extract embeddings
    imageSimilarity.create_all_vec()

    # create inverted index for searching
    imageSimilarity.calac_knn(10)

    imageSimilarity.get_knn("./geological_similarity/marble/MGN0Z.jpg", 10)
    imageSimilarity.get_knn("./geological_similarity/marble/W90SQ.jpg", 10)
    imageSimilarity.get_knn("./geological_similarity/marble/2G6SC.jpg", 10)
    imageSimilarity.get_knn("./geological_similarity/marble/PSQ1K.jpg", 10)

```
## files
- **/working/best.hdfs5** : trained weights
- **embeddings.json** : embeddings of all images
- **inverted-index.json** : inverted index
## Samples

![](/images/sample1.png)
![](/images/sample2.png)