#### this project experiment was done as part of Information, Digital, & AI Literacy course at uni.

#### this project demonstrates the complete NLP workflow in TensorFlow: preprocessing text, learning word embeddings, training a sentiment classifier, and visualizing training metrics with TensorBoard.

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