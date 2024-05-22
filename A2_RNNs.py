# %% [markdown]
# <div style="text-align: right">   </div>
# 
# 
# Introduction to Deep Learning (2023) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| &nbsp;
# -------|-------------------
# **Assignment 2 - Recurrent Neural Networks** | <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/UniversiteitLeidenLogo.svg/1280px-UniversiteitLeidenLogo.svg.png" width="300">
# 
# 
# 
# # Introduction
# 
# 
# The goal of this assignment is to learn how to use encoder-decoder recurrent neural networks (RNNs). Specifically we will be dealing with a sequence to sequence problem and try to build recurrent models that can learn the principles behind simple arithmetic operations (**integer addition, subtraction and multiplication.**).
# 
# <img src="https://i.ibb.co/5Ky5pbk/Screenshot-2023-11-10-at-07-51-21.png" alt="Screenshot-2023-11-10-at-07-51-21" border="0" width="500"></a>
# 
# In this assignment you will be working with three different kinds of models, based on input/output data modalities:
# 1. **Text-to-text**: given a text query containing two integers and an operand between them (+ or -) the model's output should be a sequence of integers that match the actual arithmetic result of this operation
# 2. **Image-to-text**: same as above, except the query is specified as a sequence of images containing individual digits and an operand.
# 3. **Text-to-image**: the query is specified in text format as in the text-to-text model, however the model's output should be a sequence of images corresponding to the correct result.
# 
# 
# ### Description**
# Let us suppose that we want to develop a neural network that learns how to add or subtract
# two integers that are at most two digits long. For example, given input strings of 5 characters: ‘81+24’ or
# ’41-89’ that consist of 2 two-digit long integers and an operand between them, the network should return a
# sequence of 3 characters: ‘105 ’ or ’-48 ’ that represent the result of their respective queries. Additionally,
# we want to build a model that generalizes well - if the network can extract the underlying principles behind
# the ’+’ and ’-’ operands and associated operations, it should not need too many training examples to generate
# valid answers to unseen queries. To represent such queries we need 13 unique characters: 10 for digits (0-9),
# 2 for the ’+’ and ’-’ operands and one for whitespaces ’ ’ used as padding.
# The example above describes a text-to-text sequence mapping scenario. However, we can also use different
# modalities of data to represent our queries or answers. For that purpose, the MNIST handwritten digit
# dataset is going to be used again, however in a slightly different format. The functions below will be used to create our datasets.
# 
# ---
# 
# *To work on this notebook you should create a copy of it.*
# 

# %% [markdown]
# # Function definitions for creating the datasets
# 
# First we need to create our datasets that are going to be used for training our models.
# 
# In order to create image queries of simple arithmetic operations such as '15+13' or '42-10' we need to create images of '+' and '-' signs using ***open-cv*** library. We will use these operand signs together with the MNIST dataset to represent the digits.

# %%
import tensorflow as tf
from keras import layers, models, optimizers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector, Dropout
# from keras.layers import RNN, Flatten, LSTMCell, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose, Conv2D, MaxPooling2D
# from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime


# Set seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# flags below should be set to run only a subset of the notebook
# addition and substraction in 3 flavors
run_text2text_53 = True
run_image2text_53 = True
run_text2image_53 = True
# multiplication
run_text2text_54 = True
run_image2text_54 = True 


# %%
from scipy.ndimage import rotate
# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):

    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates
        if sign == '*':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
            # Rotate 45 degrees
            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)

    return blank_images

def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))

# %%
# Illustrate the generated query/answer pairs
unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()


def create_data(unique_characters, highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=False)
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=False)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)


# %% [markdown]
# # Creating our data
# 
# The dataset consists of 20000 samples that (additions and subtractions between all 2-digit integers) and they have two kinds of inputs and label modalities:
# 
#   **X_text**: strings containing queries of length 5: ['  1+1  ', '11-18', ...]
# 
#   **X_image**: a stack of images representing a single query, dimensions: [5, 28, 28]
# 
#   **y_text**: strings containing answers of length 3: ['  2', '156']
# 
#   **y_image**: a stack of images that represents the answer to a query, dimensions: [3, 28, 28]

# %%
X_text, X_img, y_text, y_img = create_data(unique_characters, highest_integer) 
X_text, X_img, y_text, y_img = shuffle(X_text, X_img, y_text, y_img, random_state=42)

print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)

## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])

# %% [markdown]
# ## Helper functions
# 
# The functions below will help with input/output of the data.

# %%
# One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs
# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
  n = len(labels)
  length = len(labels[0])
  char_map = dict(zip(unique_characters, range(len(unique_characters))))
  one_hot = np.zeros([n, length, len(unique_characters)])
  for i, label in enumerate(labels):
      m = np.zeros([length, len(unique_characters)])
      for j, char in enumerate(label):
          m[j, char_map[char]] = 1
      one_hot[i] = m

  return one_hot


def decode_labels(labels):
    pred = np.argmax(labels, axis=1)
    predicted = ''.join([unique_characters[i] for i in pred])

    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

# print(X_text_onehot.shape, y_text_onehot.shape)

# %% [markdown]
# ---
# ---
# 
# ## I. Text-to-text RNN model
# 
# The following code showcases how Recurrent Neural Networks (RNNs) are built using Keras. Several new layers are going to be used:
# 
# 1. LSTM
# 2. TimeDistributed
# 3. RepeatVector
# 
# The code cell below explains each of these new components.
# 
# <img src="https://i.ibb.co/NY7FFTc/Screenshot-2023-11-10-at-09-27-25.png" alt="Screenshot-2023-11-10-at-09-27-25" border="0" width="500"></a>
# 

# %%
def build_text2text_model():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text

# %%
# Analyze the code for generating numerical and image queries and their respective answers from MNIST
# data. Inspect the provided text-to-text RNN model and try to understand the dimensionality of the
# inputs and output tensors as well as how they are encoded/decoded (one-hot format).

# Set seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

if run_text2text_53:

    def get_predictions_and_true_labels(model, X_test, y_test):
        predictions = model.predict(X_test)
        predictions = np.argmax(predictions, axis=-1)
        true_labels = np.argmax(y_test, axis=-1)
        return predictions, true_labels

    def display_misclassified_examples(X_test, y_test, model, decode_labels, num_examples=10):
        misclassified = []
        for i in range(len(X_test)):
            pred = model.predict(np.array([X_test[i]]))
            decoded_pred = decode_labels(pred[0])
            decoded_true = decode_labels(y_test[i])
            if decoded_pred.strip() != decoded_true.strip():
                misclassified.append((X_test[i], decoded_true, decoded_pred))
            if len(misclassified) >= num_examples:
                break

        print(f"Showing {num_examples} misclassified examples:")
        for example in misclassified:
            print("Input: ", decode_labels(example[0]))
            print("Actual: ", example[1].strip())
            print("Predicted: ", example[2].strip())
            print("-" * 30)

    def build_text2text_model(unique_characters, max_answer_length, learning_rate, dropout_rate):
        text2text = tf.keras.Sequential()
        text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))
        text2text.add(RepeatVector(max_answer_length))
        text2text.add(LSTM(256, return_sequences=True))
        text2text.add(Dropout(dropout_rate))  # Dropout after the second LSTM layer to reduce overfitting
        text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

        # Define the optimizer with the desired learning rate
        adam_optimizer = Adam(learning_rate=learning_rate)

        # Compile the model with the custom optimizer
        text2text.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
        text2text.summary()
        plot_model(text2text, to_file='./figures/model_text2text_plot.png', show_shapes=True, show_layer_names=True)

        return text2text


    def main(learning_rate, dropout_rate, epochs, test_size):
        unique_characters = '0123456789+- '
        highest_integer = 99
        max_answer_length = 3

        epochs=epochs
        dropout_rate=dropout_rate
        learning_rate=learning_rate
        patience_early_stopping=10 # patience: stop if train loss fails to reduce 
        patience_learning_rate=5   # patience: reduce lr if val loss fails to reduce

        X_text, X_img, y_text, y_img = create_data(highest_integer)
        X_text, X_img, y_text, y_img = shuffle(X_text, X_img, y_text, y_img, random_state=random_seed)

        X_text_onehot = encode_labels(X_text)
        y_text_onehot = encode_labels(y_text)

        # Split the dataset into a training set and a temporary set (combining validation and test) which is split the same way
        X_train, X_temp, y_train, y_temp = train_test_split(X_text_onehot, y_text_onehot, test_size=test_size, random_state=random_seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_seed)
    
        text2text_model = build_text2text_model(unique_characters, max_answer_length, learning_rate, dropout_rate)

        # Define callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience_early_stopping, verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_learning_rate, min_lr=1.e-5, verbose=1),
                    TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)]

        # Training the model with callbacks
        text2text_model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_data=(X_val, y_val), callbacks=callbacks)

        predictions, true_labels = get_predictions_and_true_labels(text2text_model, X_test, y_test)
        
        # Generate a confusion matrix
        conf_matrix = confusion_matrix(true_labels.flatten(), predictions.flatten())

        # Generate a classification report
        unique_labels = np.unique(np.concatenate([true_labels.flatten(), predictions.flatten()]))
        class_report = classification_report(true_labels.flatten(), predictions.flatten(), labels=unique_labels, target_names=[unique_characters[i] for i in unique_labels])

        print("Confusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", class_report)

        # Evaluating the model
        _, test_acc = text2text_model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc}")

        # Call this function after training and evaluating your model
        display_misclassified_examples(X_test, y_test, text2text_model, decode_labels)

        # Making a prediction
        sample_problem = encode_labels(np.array(['23+17']))
        predicted_solution = text2text_model.predict(sample_problem)
        decoded_solution = decode_labels(predicted_solution[0])
        print(f"Predicted Solution: {decoded_solution}")

        return test_acc

    if __name__ == "__main__":
        epochs=200
        test_size=0.1
        learning_rate=1.e-3
        dropout_rate=0.1

        # ceate a loop on test sizes to see performance
        test_accuracies = []
        test_sizes = np.arange(0.1, 1.0, 0.1)  
        for test_size in test_sizes:
            test_accuracy = main(learning_rate, dropout_rate, epochs, test_size)
            test_accuracies.append(test_accuracy)

        # keep results
        np.savez('./results/text2text_test_performance.npz', test_sizes=test_sizes, test_accuracies=test_accuracies)
        print("Data saved successfully.")

        # exit with a plot for inclusion in report
        plt.plot(test_sizes, test_accuracies)
        plt.scatter(test_sizes, test_accuracies, marker='*', s=50)

        # loop through each point to place a text label
        for i, txt in enumerate(test_accuracies):
            plt.text(test_sizes[i], test_accuracies[i] - 0.01, f'{txt:.2f}', ha='center', va='top')  

        plt.xlabel('test_size t')
        plt.ylabel('Test Accuracy')
        plt.ylim([0, 1])
        plt.title('text2text_53 Model Performance Across Test Sizes')
        plt.tight_layout()
        plt.savefig('./figures/experiment_1: text2text_53 accuracy across test sizes.pdf')
        plt.show()
            


# %% [markdown]
# 
# ---
# ---
# 
# ## II. Image to text RNN Model
# 
# Hint: There are two ways of building the encoder for such a model - again by using the regular LSTM cells (with flattened images as vectors) or recurrect convolutional layers [ConvLSTM2D](https://keras.io/api/layers/recurrent_layers/conv_lstm2d/).
# 
# The goal here is to use **X_img** as inputs and **y_text** as outputs.

# %%
# Create an image-to-text RNN model: given a sequence of MNIST images that represent a query of an
# arithmetic operation, your model should return the answer in text format. Once you have trained your
# model, evaluate its accuracy and compare it to the text-to-text model.

# Set seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

if run_image2text_53:

    def build_image2text_model(input_shape, unique_characters, max_answer_length, learning_rate=0.001, dropout_rate=0.1):
        image2text = models.Sequential()

        image2text.add(layers.Reshape((input_shape[0], 28, 28, 1), input_shape=input_shape))        
        # TimeDistributed Convolutional layers for feature extraction
        image2text.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
        image2text.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
        image2text.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))
        image2text.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
        image2text.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))

        # flatten the output for the LSTM layers
        image2text.add(layers.TimeDistributed(layers.Flatten()))

        # LSTM layers for understanding the sequence
        image2text.add(layers.LSTM(64, return_sequences=True))
        image2text.add(layers.LSTM(64))

        # dense layers for interpretation and output
        image2text.add(layers.Dense(64, activation='relu'))
        image2text.add(layers.Dropout(dropout_rate))
        image2text.add(layers.Dense(len(unique_characters) * max_answer_length, activation='softmax'))

        # reshape to match the output format
        image2text.add(layers.Reshape((max_answer_length, len(unique_characters))))

        # define the optimizer with the desired learning rate
        adam_optimizer = optimizers.Adam(learning_rate=learning_rate)

        # compile the model with the custom optimizer
        image2text.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        image2text.summary()
        # plot model for inclusion in report
        plot_model(image2text, to_file='./figures/model_text2text_plot.png', show_shapes=True, show_layer_names=True)
        return image2text

    # alternative model - tried but did not yield superior results
    # def build_convLSTM_dense_image2text_model(input_shape, unique_characters, max_answer_length, learning_rate=0.001, dropout_rate=0.1):
    #     model = models.Sequential()

    #     # Adjusted pooling sizes
    #     pooling_size_3 = (1, 3, 3)  
    #     pooling_size_2 = (1, 2, 2)  
    #     pooling_size_1 = (1, 1, 2)  # Reduced pooling size for later layers

    #     # ConvLSTM2D layers for spatiotemporal feature extraction
    #     model.add(layers.ConvLSTM2D(32, (5, 5), padding='same', return_sequences=True, input_shape=input_shape))
    #     model.add(layers.BatchNormalization())
    #     model.add(layers.MaxPooling3D(pool_size=pooling_size_3))  

    #     model.add(layers.ConvLSTM2D(32, (5, 5), padding='same', return_sequences=True))
    #     model.add(layers.BatchNormalization())
    #     model.add(layers.MaxPooling3D(pool_size=pooling_size_2))  

    #     model.add(layers.ConvLSTM2D(32, (5, 5), padding='same', return_sequences=True))
    #     model.add(layers.BatchNormalization())
    #     model.add(layers.MaxPooling3D(pool_size=pooling_size_1))

    #     model.add(layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=False))
    #     model.add(layers.BatchNormalization())

    #     # Flatten the output for the Dense layers
    #     model.add(layers.Flatten())

    #     # Dense layers for interpretation and output
    #     model.add(layers.Dense(64, activation='relu'))
    #     model.add(layers.Dropout(dropout_rate))
    #     model.add(layers.Dense(len(unique_characters) * max_answer_length, activation='softmax'))

    #     # Reshape to match the output format
    #     model.add(layers.Reshape((max_answer_length, len(unique_characters))))

    #     # Define the optimizer with the desired learning rate
    #     adam_optimizer = optimizers.Adam(learning_rate=learning_rate)

    #     # Compile the model with the custom optimizer
    #     model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #     model.summary()

    #     return model

    def main(learning_rate, dropout_rate, epochs, test_size):
        unique_characters = '0123456789+- '
        highest_integer = 99
        max_answer_length = 3

        random_seed=42
        epochs=epochs
        dropout_rate=dropout_rate
        learning_rate=learning_rate
        patience_early_stopping=50
        patience_learning_rate=5

        # Generate the dataset
        X_text, X_img, y_text, y_img = create_data(unique_characters, highest_integer)
        X_text, X_img, y_text, y_img = shuffle(X_text, X_img, y_text, y_img, random_state=random_seed)

        # Preprocess images for LSTM
        X_img_preprocessed = X_img.reshape(X_img.shape[0], X_img.shape[1], 28, 28, 1)
        y_text_onehot = encode_labels(y_text)  # Implement encode_labels accordingly

        # Splitting the dataset
        X_train, X_temp, y_train, y_temp = train_test_split(X_img_preprocessed, y_text_onehot, test_size=test_size, random_state=random_seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_seed)

        # Building the model
        model = build_image2text_model(X_train.shape[1:], unique_characters, max_answer_length, learning_rate, dropout_rate)

        # Define callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience_early_stopping, verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_learning_rate, min_lr=1.e-5, verbose=1),
                    TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)]

        history=model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_data=(X_val, y_val), callbacks=callbacks)

        _, test_acc = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc}")
        np.save('./results/image2text_training_history.npy', history.history)
        model.save('./models/image2text_model.h5')

        return test_acc

    
    if __name__ == "__main__":

        epochs=400
        learning_rate=1.e-3
        dropout_rate=0.2
        unique_characters = '0123456789+- '

        test_sizes = np.arange(0.1, 1.0, 0.2)  
        for test_size in test_sizes:
            test_accuracy=main(learning_rate, dropout_rate, epochs, test_size)
                
            np.savez(f'./results/image2text_53_test_performance_{test_size}.npz', 
                test_sizes=test_sizes, 
                test_accuracies=test_accuracy)
                
            print("Data saved successfully.")

        plt.plot(test_sizes, test_accuracy)
        plt.scatter(test_sizes, test_accuracy, marker='*', s=50)
        for i, txt in enumerate(test_accuracy):
            plt.text(test_sizes[i], test_accuracy[i], f'{txt:.4f}', ha='center', va='top')  

        plt.xlabel('test_size t')
        plt.ylabel('Test Accuracy')
        plt.ylim([0, 1])
        plt.title('image2text_53 model performance across different test sizes')
        plt.tight_layout()
        plt.savefig('./figures/experiment_2 image2text_53 accuracy.pdf')
        plt.show()

# %%
# visualize the results of training (not used in report)

if run_image2text_53:

    # Load the training history
    history = np.load('./results/image2text_training_history.npy', allow_pickle='TRUE').item()

    # Plot the training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.yscale('log')
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5')  # Major grid for both axes
    plt.grid(which='minor', axis='y', linestyle=':', linewidth='0.5')  # Minor grid for y-axis
    plt.minorticks_on()  # Enable minor ticks on y-axis
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5')  # Major grid for both axes
    plt.grid(which='minor', axis='y', linestyle=':', linewidth='0.5')  # Minor grid for y-axis
    plt.minorticks_on()  # Enable minor ticks on y-axis
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/Fig_3_learning_curves.pdf')
    plt.show()

    # Summarize Performance Statistics
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_accuracy = history['accuracy'][-1]
    final_val_accuracy = history['val_accuracy'][-1]

    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")


# %% [markdown]
# ---
# ---
# 
# ## III. Text to image RNN Model
# 
# Hint: to make this model work really well you could use deconvolutional layers in your decoder (you might need to look up ***Conv2DTranspose*** layer). However, regular vector-based decoder will work as well.
# 
# The goal here is to use **X_text** as inputs and **y_img** as outputs.

# %%
# Build a text-to-image RNN model: given a text query, your network should generate a sequence of
# images that represent the correct answer. In this case, it is harder to evaluate the performance of your
# model qualitatively. However, you should provide examples of the output generated by your model in
# the report. What can you say about the appearance of these generated images?

if run_text2image_53:
       
    def classify_digits(mnist_model, image):
        # Preprocess the image (reshape, normalize, etc.) as done for MNIST training
        # Ensure the image is reshaped to (1, 28, 28, 1)
        image_preprocessed = image.reshape(1, 28, 28, 1) / 255.0
        # Predict the digit class
        digit_prediction = mnist_model.predict(image_preprocessed)
        return np.argmax(digit_prediction, axis=1)

    def calculate_accuracy(true_labels, predicted_labels):
        correct = np.sum(true_labels == predicted_labels)
        total = len(true_labels)
        return correct / total

    def create_data(unique_characters, highest_integer, num_addends=2, operands=['+', '-']):
        """
        Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

        @return:
        X_text: '51+21' -> text query of an arithmetic operation (5)
        X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
        y_text: '72' -> answer of the arithmetic text query
        y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

        Images for digits are picked randomly from the whole MNIST dataset.
        """
        (MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
        max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
        max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
        max_answer_length = max_int_length + 1    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')
        num_indices = [np.where(MNIST_labels==x) for x in range(10)]
        num_data = [MNIST_data[inds] for inds in num_indices]
        
        image_mapping = dict(zip(unique_characters[:10], num_data))
        image_mapping['-'] = generate_images()
        image_mapping['+'] = generate_images(sign='+')
        image_mapping['*'] = generate_images(sign='*')
        image_mapping[' '] = np.zeros([1, 28, 28])

        X_text, X_img, y_text, y_img = [], [], [], []

        for i in range(highest_integer + 1):      # First addend
            for j in range(highest_integer + 1):  # Second addend
                for sign in operands: # Create all possible combinations of operands
                    query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=False)
                    query_image = []
                    for n, char in enumerate(query_string):
                        image_set = image_mapping[char]
                        index = np.random.randint(0, len(image_set), 1)
                        query_image.append(image_set[index].squeeze())

                    result = eval(query_string)
                    result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=False)
                    result_image = []
                    for n, char in enumerate(result_string):
                        image_set = image_mapping[char]
                        index = np.random.randint(0, len(image_set), 1)
                        result_image.append(image_set[index].squeeze())

                    X_text.append(query_string)
                    X_img.append(np.stack(query_image))
                    y_text.append(result_string)
                    y_img.append(np.stack(result_image))

        return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

    def display_samples(X_text, y_true, y_pred, sample_indices):
        plt.figure(figsize=(5, 10))  

        for i, idx in enumerate(sample_indices):
            # Display true images
            plt.subplot(10, 2, 2*i + 1)
            plt.imshow(np.hstack(y_true[idx]), cmap='gray')
            plt.title(f"True Images for {i+1}:  {str(X_text[idx])}")
            plt.axis('off')

            # Display predicted images
            plt.subplot(10, 2, 2*i + 2)
            plt.imshow(np.hstack(y_pred[idx].squeeze()), cmap='gray')
            plt.title(f"Predicted Images for {i+1}:  {str(X_text[idx])}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'./figures/text2image_53_ts{test_size:2.1f}_ep{epochs}_nl{num_layers}_final.pdf')
        plt.show()

    def build_text2image_model(unique_characters, digit_image_shape, learning_rate, dropout_rate, num_lstm_layers):
        # Text processing part (RNN)
        text_input = layers.Input(shape=(None, len(unique_characters)))
        x = layers.LSTM(256, return_sequences=True)(text_input)
        x = layers.LSTM(256)(x)
        x = layers.Dropout(dropout_rate)(x)

        # Image generation part
        num_segments = 3  # Number of images in the sequence
        segment_outputs = []
        for _ in range(num_segments):
            segment = layers.Dense(128, activation='relu')(x)
            segment = layers.Dense(np.prod(digit_image_shape), activation='sigmoid')(segment)
            segment = layers.Reshape(digit_image_shape)(segment)
            segment_outputs.append(segment)

        # Output the segments as a sequence
        image_output = layers.Lambda(lambda x: tf.stack(x, axis=1))(segment_outputs)

        # Build and compile the model
        model = models.Model(inputs=text_input, outputs=image_output)
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
        model.summary()
        plot_model(model, to_file='./figures/model_text2image_plot.png', show_shapes=True, show_layer_names=True)
        return model


    def main(learning_rate, dropout_rate, epochs, test_size):

        # data initialization
        unique_characters = '0123456789+- '
        highest_integer = 99

        # architecture
        digit_image_shape = (28, 28, 1)  # MNIST image shape

        # hyper parameters
        random_seed=42
        epochs=epochs
        dropout_rate=dropout_rate
        learning_rate=learning_rate
        patience_early_stopping=50
        patience_learning_rate=5
        
        # Generate the dataset
        X_text, X_img, y_text, y_img = create_data(unique_characters, highest_integer, num_addends=2, operands=['+', '-'])
        X_text, X_img, y_text, y_img = shuffle(X_text, X_img, y_text, y_img, random_state=random_seed)
        print(X_text.shape, y_img.shape)
        # Preprocess images for LSTM
        X_text_onehot = encode_labels(X_text)
        y_img_preprocessed = y_img.reshape(y_img.shape[0], y_img.shape[1], 28, 28)
        print(X_text_onehot.shape, y_img_preprocessed.shape)

        # Splitting the dataset
        X_train_onehot, X_temp_onehot, y_train, y_temp, X_train_text, X_temp_text = train_test_split(
            X_text_onehot, y_img_preprocessed, X_text, test_size=test_size, random_state=random_seed)

        X_val_onehot, X_test_onehot, y_val, y_test, X_val_text, X_test_text = train_test_split(
            X_temp_onehot, y_temp, X_temp_text, test_size=test_size, random_state=random_seed)
    
        model = build_text2image_model(unique_characters, digit_image_shape, learning_rate, dropout_rate, num_layers)
        
        # Define callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience_early_stopping, verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_learning_rate, min_lr=1.e-5, verbose=1),
                    TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)]

        history = model.fit(X_train_onehot, y_train, epochs=epochs, batch_size=128, validation_data=(X_val_onehot, y_val), callbacks=callbacks)        
        model.save(f'./models/model_text2image_53_new_e{epochs}_ts{test_size:2.1f}.h5')

        # Generate predictions for the test set
        predicted_test = model.predict(X_test_onehot)

        # Select 10 random test samples
        sample_indices = np.random.choice(len(X_test_text), 10, replace=False)

        # Display the selected samples
        display_samples(X_test_text, y_test, predicted_test, sample_indices)


    if __name__ == "__main__":

        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        epochs=200
        learning_rate=1.e-3
        dropout_rate=0.03
        unique_characters = '0123456789+- '
        highest_integer = 99
        num_layers=3

        test_accuracies = []
        test_sizes = np.arange(0.1, 1.0, 0.2)  

        for test_size in test_sizes:
            main(learning_rate, dropout_rate, epochs, test_size)
 
 

# %% [markdown]
# 

# %%
# Try adding additional LSTM layers to your encoder networks and see how the performance of your
# models changes. Try to explain these performance differences in the context of the mistakes that
# your network was making before. Tip: you should add a flag ”return sequences=True” to the first
# recurrent layer of your network.

# %% [markdown]
# 
# ---
# ---
# ---
# 
# # Part 2: Multiplication
# The cell below will create the multiplication dataset used in this part of the assignment.

# %%
# Illustrate the generated query/answer pairs

# Now try building models around the multiplication dataset (last cell of the notebook) - one that
# contains all combinations of two-digit integer multiplications (10,000 samples).

if run_text2text_54 or run_image2text_54:
    unique_characters = '0123456789* '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
    highest_integer = 99                      # Highest value of integers contained in the queries

    max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
    max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
    max_answer_length = 5    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

    # Create the data (might take around a minute)
    (MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
    X_text, X_img, y_text, y_img = create_data(unique_characters, highest_integer, operands=['*'])
    print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


    ## Display the samples that were created
    def display_sample(n):
        labels = ['X_img:', 'y_img:']
        for i, data in enumerate([X_img, y_img]):
            plt.subplot(1,2,i+1)
            # plt.set_figheight(15)
            plt.axis('off')
            plt.title(labels[i])
            plt.imshow(np.hstack(data[n]), cmap='gray')
        print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
        plt.show()

    for _ in range(10):
        display_sample(np.random.randint(0, 10000, 1)[0])




# %%
# Create text-to-text and image-to-text models for the multiplication problem. How do the generalization
# capabilities change compared to Task 1?

# Analyze the code for generating numerical and image queries and their respective answers from MNIST
# data. Inspect the provided text-to-text RNN model and try to understand the dimensionality of the
# inputs and output tensors as well as how they are encoded/decoded (one-hot format).

# Set seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

if run_text2text_54:
    def get_predictions_and_true_labels(model, X_test, y_test):
        predictions = model.predict(X_test)
        predictions = np.argmax(predictions, axis=-1)
        true_labels = np.argmax(y_test, axis=-1)
        return predictions, true_labels

    def display_misclassified_examples(X_test, y_test, model, decode_labels, num_examples=10):
        misclassified = []
        for i in range(len(X_test)):
            pred = model.predict(np.array([X_test[i]]))
            decoded_pred = decode_labels(pred[0])
            decoded_true = decode_labels(y_test[i])
            if decoded_pred.strip() != decoded_true.strip():
                misclassified.append((X_test[i], decoded_true, decoded_pred))
            if len(misclassified) >= num_examples:
                break

        print(f"Showing {num_examples} misclassified examples:")
        for example in misclassified:
            print("Input: ", decode_labels(example[0]))
            print("Actual: ", example[1].strip())
            print("Predicted: ", example[2].strip())
            print("-" * 30)

    def build_text2text_model(unique_characters, max_answer_length, learning_rate, dropout_rate):
        text2text = tf.keras.Sequential()
        text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))
        text2text.add(RepeatVector(max_answer_length))
        text2text.add(LSTM(256, return_sequences=True))
        text2text.add(Dropout(dropout_rate))  # Dropout after the second LSTM layer
        text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

        # Define the optimizer with the desired learning rate
        adam_optimizer = Adam(learning_rate=learning_rate)

        # Compile the model with the custom optimizer
        text2text.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
        text2text.summary()

        return text2text


    def main(learning_rate, dropout_rate, epochs, test_size):

        epochs=epochs
        dropout_rate=dropout_rate
        learning_rate=learning_rate
        patience_early_stopping=10
        patience_learning_rate=5

        X_text, X_img, y_text, y_img = create_data(unique_characters, highest_integer, operands=['*'])

        X_text_onehot = encode_labels(X_text)
        y_text_onehot = encode_labels(y_text)

        # Splitting the dataset
        # Split the dataset into a training set and a temporary set (combining validation and test)
        X_train, X_temp, y_train, y_temp = train_test_split(X_text_onehot, y_text_onehot, test_size=test_size, random_state=random_seed)

        # Split the temporary set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_seed)

        text2text_model = build_text2text_model(unique_characters, max_answer_length, learning_rate, dropout_rate)

        # Define callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience_early_stopping, verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_learning_rate, min_lr=1.e-5, verbose=1),
                    TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)]

        # Training the model with callbacks
        text2text_model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_data=(X_val, y_val), callbacks=callbacks)

        predictions, true_labels = get_predictions_and_true_labels(text2text_model, X_test, y_test)
        
        # Generate a confusion matrix
        conf_matrix = confusion_matrix(true_labels.flatten(), predictions.flatten())

        # Generate a classification report
        unique_labels = np.unique(np.concatenate([true_labels.flatten(), predictions.flatten()]))
        class_report = classification_report(true_labels.flatten(), predictions.flatten(), labels=unique_labels, target_names=[unique_characters[i] for i in unique_labels])

        print("Confusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", class_report)

        # Evaluating the model
        test_loss, test_acc = text2text_model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc}")

        # Call this function after training and evaluating your model
        display_misclassified_examples(X_test, y_test, text2text_model, decode_labels)

        # Making a prediction
        sample_problem = encode_labels(np.array(['23*17']))
        predicted_solution = text2text_model.predict(sample_problem)
        decoded_solution = decode_labels(predicted_solution[0])
        print(f"Predicted Solution: {decoded_solution}")

        return test_acc

    if __name__ == "__main__":
        # hyper parameters
        epochs=500
        learning_rate=1.e-3
        dropout_rate=0.1
        # problem parameters
        unique_characters = '0123456789* '
        highest_integer = 99
        max_query_length = len(str(highest_integer)) * 2 + 1
        max_answer_length = len(str(highest_integer)) * 2

        test_accuracies = []
        test_sizes = np.arange(0.1, 1.0, 0.1)  # Creates an array from 0.1 to 0.9 with a step of 0.1
        for test_size in test_sizes:
            test_accuracy = main(learning_rate, dropout_rate, epochs, test_size)
            test_accuracies.append(test_accuracy)

        # Assuming test_sizes and test_accuracies are defined and contain your data
        np.savez('./results/image2text_54_test_performance.npz', test_sizes=test_sizes, test_accuracies=test_accuracies)
        print("Data saved successfully.")

        plt.plot(test_sizes, test_accuracies)
        plt.xlabel('Test Size')
        plt.ylabel('Test Accuracy')
        plt.title('text2text model performance across different test sizes')
        plt.tight_layout()
        plt.savefig('./figures/experiment_6: image2text accuracy across test sizes.pdf')
        plt.show()
            


# %%
if run_image2text_54:

    import tensorflow as tf
    from keras import layers, models, optimizers
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.utils import shuffle
    from keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell
    from keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose
    from keras.layers import Conv2D, MaxPooling2D, Reshape, Dropout
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.utils import plot_model
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
    from sklearn.model_selection import train_test_split
    from scipy.ndimage import rotate
    import datetime


    # Set seeds for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Create plus/minus operand signs
    def generate_images(number_of_images=50, sign='-'):

        blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
        x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
        y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
        y2 = np.random.randint(18, 22, number_of_images)     # -||-

        for i in range(number_of_images): # Generate n different images
            cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
            if sign == '+':
                cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates
            if sign == '*':
                cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
                # Rotate 45 degrees
                blank_images[i] = rotate(blank_images[i], -50, reshape=False)
                cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
                blank_images[i] = rotate(blank_images[i], -50, reshape=False)
                cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)

        return blank_images

    def show_generated(images, n=5):
        plt.figure(figsize=(2, 2))
        for i in range(n**2):
            plt.subplot(n, n, i+1)
            plt.axis('off')
            plt.imshow(images[i])
        plt.show()

    # unique_characters = '0123456789* '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
    # highest_integer = 99                      # Highest value of integers contained in the queries

    # max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
    # max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
    # max_answer_length = 5    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')


    # Create the data (might take around a minute)
    # (MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()

    def create_data(unique_characters, highest_integer, num_addends=2, operands=['+', '-']):
        """
        Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

        @return:
        X_text: '51+21' -> text query of an arithmetic operation (5)
        X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
        y_text: '72' -> answer of the arithmetic text query
        y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

        Images for digits are picked randomly from the whole MNIST dataset.
        """
        (MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
        max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
        max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
        max_answer_length = 4    # Maximum length of the answer string (the longest resulting query string is ' 99*99'='9801')

        num_indices = [np.where(MNIST_labels==x) for x in range(10)]
        num_data = [MNIST_data[inds] for inds in num_indices]
        image_mapping = dict(zip(unique_characters[:10], num_data))
        image_mapping['-'] = generate_images()
        image_mapping['+'] = generate_images(sign='+')
        image_mapping['*'] = generate_images(sign='*')
        image_mapping[' '] = np.zeros([1, 28, 28])

        X_text, X_img, y_text, y_img = [], [], [], []

        for i in range(highest_integer + 1):      # First addend
            for j in range(highest_integer + 1):  # Second addend
                for sign in operands: # Create all possible combinations of operands
                    query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=False)
                    query_image = []
                    for n, char in enumerate(query_string):
                        image_set = image_mapping[char]
                        index = np.random.randint(0, len(image_set), 1)
                        query_image.append(image_set[index].squeeze())

                    result = eval(query_string)
                    result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=False)
                    result_image = []
                    for n, char in enumerate(result_string):
                        image_set = image_mapping[char]
                        index = np.random.randint(0, len(image_set), 1)
                        result_image.append(image_set[index].squeeze())

                    X_text.append(query_string)
                    X_img.append(np.stack(query_image))
                    y_text.append(result_string)
                    y_img.append(np.stack(result_image))

        return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

    def to_padded_chars(integer, max_len=3, pad_right=False):
        """
        Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
        """
        length = len(str(integer))
        padding = (max_len - length) * ' '
        if pad_right:
            return str(integer) + padding
        else:
            return padding + str(integer)

    def encode_labels(labels, max_len=3):
        n = len(labels)
        length = len(labels[0])
        char_map = dict(zip(unique_characters, range(len(unique_characters))))
        one_hot = np.zeros([n, length, len(unique_characters)])
        for i, label in enumerate(labels):
            m = np.zeros([length, len(unique_characters)])
            for j, char in enumerate(label):
                m[j, char_map[char]] = 1
            one_hot[i] = m

        return one_hot

    def decode_labels(labels):
        pred = np.argmax(labels, axis=1)
        predicted = ''.join([unique_characters[i] for i in pred])

        return predicted

    def get_predictions_and_true_labels(model, X_test, y_test):
        predictions = model.predict(X_test)
        predictions = np.argmax(predictions, axis=-1)
        true_labels = np.argmax(y_test, axis=-1)
        return predictions, true_labels

    def display_misclassified_examples(X_test, y_test, model, decode_labels, num_examples=10):
        misclassified = []
        for i in range(len(X_test)):
            pred = model.predict(np.array([X_test[i]]))
            decoded_pred = decode_labels(pred[0])
            decoded_true = decode_labels(y_test[i])
            if decoded_pred.strip() != decoded_true.strip():
                misclassified.append((X_test[i], decoded_true, decoded_pred))
            if len(misclassified) >= num_examples:
                break

        print(f"Showing {num_examples} misclassified examples:")
        for example in misclassified:
            print("Input: ", decode_labels(example[0]))
            print("Actual: ", example[1].strip())
            print("Predicted: ", example[2].strip())
            print("-" * 30)

    def build_image2text_model(input_shape, unique_characters, max_answer_length, learning_rate=0.001, dropout_rate=0.1):
        image2text = models.Sequential()

        image2text.add(layers.Reshape((input_shape[0], 28, 28, 1), input_shape=input_shape))        
        # TimeDistributed Convolutional layers for feature extraction
        image2text.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
        image2text.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
        image2text.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))
        image2text.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
        image2text.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))

        # Flatten the output for the LSTM layers
        image2text.add(layers.TimeDistributed(layers.Flatten()))

        # LSTM layers for understanding the sequence
        image2text.add(layers.LSTM(64, return_sequences=True))
        image2text.add(layers.LSTM(64))

        # Dense layers for interpretation and output
        image2text.add(layers.Dense(64, activation='relu'))
        image2text.add(layers.Dropout(dropout_rate))
        image2text.add(layers.Dense(len(unique_characters) * max_answer_length, activation='softmax'))

        # Reshape to match the output format
        image2text.add(layers.Reshape((max_answer_length, len(unique_characters))))

        # Define the optimizer with the desired learning rate
        adam_optimizer = optimizers.Adam(learning_rate=learning_rate)

        # Compile the model with the custom optimizer
        image2text.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        image2text.summary()
        plot_model(image2text, to_file='./figures/model_image2text_54_plot.png', show_shapes=True, show_layer_names=True)

        return image2text


    def main(learning_rate, dropout_rate, epochs, test_size):

        highest_integer = 99
        max_answer_length = 4

        epochs=epochs
        dropout_rate=dropout_rate
        learning_rate=learning_rate
        patience_early_stopping=30
        patience_learning_rate=10

        # Generate the dataset
        X_text, X_img, y_text, y_img = create_data(unique_characters, highest_integer, num_addends=2, operands='*')
        X_text, X_img, y_text, y_img = shuffle(X_text, X_img, y_text, y_img, random_state=random_seed)

        # Preprocess images for LSTM
        X_img_preprocessed = X_img.reshape(X_img.shape[0], X_img.shape[1], 28, 28, 1)
        y_text_onehot = encode_labels(y_text)  # Implement encode_labels accordingly

        # Splitting the dataset
        X_train, X_temp, y_train, y_temp = train_test_split(X_img_preprocessed, y_text_onehot, test_size=test_size, random_state=random_seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_seed)

        # Building the model
        model = build_image2text_model(X_train.shape[1:], unique_characters, max_answer_length, learning_rate, dropout_rate)

        # Define callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience_early_stopping, verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_learning_rate, min_lr=1.e-6, verbose=1),
                    TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)]

        history=model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_data=(X_val, y_val), callbacks=callbacks)

        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc}")
        np.save('./results/image2text_54_training_history.npy', history.history)
        model.save('./models/image2text_54_model.h5')

        return test_acc

    if __name__ == "__main__":
        epochs=1000 # epochs=500 is used in the report, level is increased to attempt better performance
        learning_rate=1.e-4
        dropout_rate=0.1 #dropout of 0.05 is shown in the report, level is increased to attempt to reduce overfitting
        unique_characters = '0123456789* '
        highest_integer = 99
        
        test_accuracies = []
        test_sizes = np.arange(0.1, 0.2, 0.2)  
        for test_size in test_sizes:
            print(f"test size {test_size:2.1f}")
            test_accuracy = main(learning_rate, dropout_rate, epochs, test_size)
            test_accuracies.append(test_accuracy)

        # Assuming test_sizes and test_accuracies are defined and contain your data
        np.savez('./results/image2text_54_test_performance.npz', test_sizes=test_sizes, test_accuracies=test_accuracies)
        print("Data saved successfully.")

        plt.plot(test_sizes, test_accuracies)
        plt.scatter(test_sizes, test_accuracies, marker='*', s=50)

        # Loop through each point to place a text label
        for i, txt in enumerate(test_accuracies):
            plt.text(test_sizes[i], test_accuracies[i] - 0.01, f'{txt:.2f}', ha='center', va='top')  

        plt.xlabel('test_size t')
        plt.ylabel('Test Accuracy')
        plt.ylim([0, 1])
        plt.title('image2text_54 model performance across test sizes')
        plt.tight_layout()
        plt.savefig(f'./figures/experiment_6: image2text_54 accuracy across test sizes {epochs} {dropout_rate} {learning_rate}.pdf')
    # experiment 3: t2t addition/substraction


# %%
# Apply what you learned in the previous tasks and try to tune the architecture/hyperparameters of
# your models further. What is the highest accuracy on the test set that you can achieve and what kind
# of train/test sample ratio can you use?



