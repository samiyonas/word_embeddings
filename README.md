#### this project experiment was done as part of Information, Digital, & AI Literacy course at SET.

#### i used this guide: https://www.tensorflow.org/text/guide/word_embeddings

#### this project explores Natural Language Processing using TensorFlow’s word embeddings, following the official TensorFlow Word Embeddings guide. it builds a sentiment analysis model on the IMDb dataset using TextVectorization and an embedding layer to convert text into learnable numerical representations, followed by a neural network for binary classification. The project also uses TensorBoard to visualize training performance and understand model behavior, including learning patterns and overfitting.

![epoch accuracy](assets/epoch%20accuracy.png)
![epoch loss](assets/epoch%20loss.png)
#### the model achieved high training accuracy, but validation loss increased significantly after the first few epochs. this suggests the network began overfitting to the training dataset rather than learning more general language patterns.

![model summary](assets/model%20summary.png)

### try it yourself

#### install uv
```
pip install uv
```
#### install dependencies
```
uv sync
```
#### run it
```
uv run python3 main.py
```
#### run the tensorboard(visualization)
```
uv run tensorboard --logdir logs
```

#### to visualize the word embeddings, upload them to vectors.tsv and metadata.tsv file to https://projector.tensorflow.org