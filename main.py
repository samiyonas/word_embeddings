import io
import os
import re
import shutil
import string
import tensorflow as tf
import datetime

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

# show what's in the downloaded dataset
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb_v1_extracted/aclImdb')
print(os.listdir(dataset_dir))

# show what's in "train" directory
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

# delete unsup directory
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)


"""
turn the dataset into tf.data.Dataset data pipeline system.
it represents a sequence of data sample that tensorflow can efficiently
load, transform, batch, shuffle, and stream into a model


by using tf.data.Dataset we can efficiently feed data to our model and reduces
manual work.
"""

batch_size = 10
seed = 12
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb_v1_extracted/aclImdb/train',
    batch_size=batch_size,
    subset='training',
    seed=seed,
    validation_split=0.2
).cache().prefetch(AUTOTUNE)

val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb_v1_extracted/aclImdb/train',
    batch_size=batch_size,
    subset='validation',
    seed=seed,
    validation_split=0.2
).cache().prefetch(AUTOTUNE)


# Embed a 1,000 word vocabulary into 5 dimensions.(function)
embedding_layer = tf.keras.layers.Embedding(1000, 5)

# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', '')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

# Vocabulary size and number of words in a sequence
# maximum number of unique tokenID
vocab_size = 10000
# sequence_length = number of tokens per sentence (fixed size)
# if the sentence length is >100, it will be reduced to 100. if it is <100, padded with 0
sequence_length = 100

"""
it builds vocabulary and assigns consistent integer IDs, it doesn't learn meaning.(function)
"""
verctorize_layer = TextVectorization(
    standardize=custom_standardization,
    # we need maximum token of vocab_size(unique tokens)
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
verctorize_layer.adapt(text_ds)

em_dim= 16

# here we talk number to number: tokenID -> vector
model = Sequential([
    # (batch_size, sequence_length)
    verctorize_layer,
    # each token(representative of a word) of a sentence -> em_dim width vector.
    # (batch_size, sequence_length) -> (batch_size, sequence_length, em_dim)
    # SEMANTIC RELATIONSHIPS ARE LEARNED THROUGH BACKPROPAGATION.
    Embedding(vocab_size, em_dim, name='embedding'),
    # pooling creates a rough “meaning summary” of the sentence.
    # averages(ignores order) all token embeddings in each sequence to produce one fixed size vector
    # representing the whole sentence
    # (batch_size, sequence_length, em_dim) -> (batch_size, em_dim)
    GlobalAveragePooling1D(),
    # learning happens and relu helps the network learn complex patterns.
    # fully connected hidden layer.
    # learns higher level pattern from the pooled representation.
    Dense(16, activation='relu'),
    # final prediction layer for binary classification/two classes
    Dense(1)
])

# visualize metrics including loss and accuracy.
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

model.compile(
    # how the model learns(how it adjusts weights of the vectors)
    optimizer='adam',
    # The loss function measures: how wrong the model is
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # during training, show me percentage of correct predictions
    metrics=['accuracy']
)

model.fit(
    train_ds,
    # After each epoch, evaluate the model on unseen validation data
    # Is the model actually generalizing? instead of just memorizing training data.
    validation_data=val_ds,
    # epoch means one complete pass through the training dataset.
    # the model trains through the entire dataset 15 times.
    epochs=15,
    # extra action performed during training
    callbacks=[tensorboard_callback]
)

model.summary()

# extract the learned word embeddings from the embedding layer
# weights = (vocab_size, em_dim)
weights = model.get_layer(name='embedding').get_weights()[0]
# retrieves the vocabulary learned by TextVectorization
# return the "word -> integer mapping" as a list
vocab = verctorize_layer.get_vocabulary()

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
    if index == 0:
        continue
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + '\n')
    out_m.write(word + '\n')

out_v.close()
out_m.close()