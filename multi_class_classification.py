import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import pandas as pd

from keras.layers import Embedding, GlobalAveragePooling1D, Dropout, Dense, TextVectorization, Activation
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Model
from keras.callbacks import EarlyStopping

# region Process Data
batch_size = 32
seed = 42

train_dataset = tf.keras.utils.text_dataset_from_directory(
    'E:\\Development Projects\\AI Data\\StackOverflow\\train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

for text_batch, label_batch in train_dataset.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

print('\n')
print("Label 0 corresponds to", train_dataset.class_names[0])
print("Label 1 corresponds to", train_dataset.class_names[1])
print("Label 2 corresponds to", train_dataset.class_names[2])
print("Label 3 corresponds to", train_dataset.class_names[3])
print('\n')

validation_dataset = tf.keras.utils.text_dataset_from_directory(
    'E:\\Development Projects\\AI Data\\StackOverflow\\train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

test_dataset = tf.keras.utils.text_dataset_from_directory(
    'E:\\Development Projects\\AI Data\\StackOverflow\\test',
    batch_size=batch_size)
# endregion

# region Standardize, Tokenize and Vectorize
"""Standardization refers to preprocessing the text, typically to remove punctuation or HTML elements to simplify the 
dataset. Tokenization refers to splitting strings into tokens (for example, splitting a sentence into individual 
words, by splitting on whitespace). Vectorization refers to converting tokens into numbers so they can be fed into a 
neural network. """


def CustomStandardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(
    standardize=CustomStandardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = train_dataset.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def VectorizeText(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(train_dataset))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", train_dataset.class_names[first_label])
print("Vectorized review", VectorizeText(first_review, first_label))

print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print("313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
# endregion

# region Apply text vectorization the datasets
train_data = train_dataset.map(VectorizeText)
val_data = validation_dataset.map(VectorizeText)
test_data = test_dataset.map(VectorizeText)
# endregion

# region Configure data for performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)
# endregion

# region Create the model
embedding_dim = 16

# embed_layer = Embedding(max_features + 1, embedding_dim)
# dropout_layer_1 = Dropout(0.2)(embed_layer)
# pooling_layer = GlobalAveragePooling1D()(dropout_layer_1)
# dropout_layer_2 = Dropout(0.2)(pooling_layer)
# hidden_layer = Dense(units=4)(dropout_layer_2)

model = tf.keras.Sequential([
  Embedding(max_features + 1, embedding_dim),
  Dropout(0.2),
  GlobalAveragePooling1D(),
  Dropout(0.2),
  Dense(units=4, activation='softmax')])

model.summary()

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
# endregion

early_stop = EarlyStopping(patience=10, mode='min', restore_best_weights=True)

# region Train and evaluate the model
epochs = 20
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop])

loss, accuracy = model.evaluate(test_data)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
# endregion

# region Plot the graph
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
# endregion


# Retrain with activation and vector layer
optimized_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  Activation('softmax')
])

optimized_model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = optimized_model.evaluate(test_dataset)
print(accuracy)