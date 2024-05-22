# experiment 1: t2t addition/substraction
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
    plot_model(text2text, to_file='./figures/model_text2text_53_plot.png', show_shapes=True, show_layer_names=True)

    return text2text

def main(learning_rate, dropout_rate, epochs, test_size):
    unique_characters = '0123456789+- '
    highest_integer = 99
    max_answer_length = 3

    random_seed=42
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    epochs=epochs
    dropout_rate=dropout_rate
    learning_rate=learning_rate
    patience_early_stopping=20
    patience_learning_rate=5

    X_text, X_img, y_text, y_img = create_data(unique_characters, highest_integer)
    X_text, X_img, y_text, y_img = shuffle(X_text, X_img, y_text, y_img, random_state=random_seed)

    X_text_onehot = encode_labels(X_text)
    y_text_onehot = encode_labels(y_text)
    print(X_text_onehot.shape, y_text_onehot.shape)

    # Split the dataset 
    X_train, X_temp, y_train, y_temp = train_test_split(X_text_onehot, y_text_onehot, test_size=test_size, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_seed)
    print(np.shape(X_train))
    print(np.shape(y_train))
    

    text2text_model = build_text2text_model(unique_characters, max_answer_length, learning_rate, dropout_rate)
    text2text_model.summary()

    # Define callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience_early_stopping, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_learning_rate, min_lr=1.e-6, verbose=1),
                TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)]

    # Training the model with callbacks
    text2text_model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_data=(X_val, y_val), callbacks=callbacks)

    predictions, true_labels = get_predictions_and_true_labels(text2text_model, X_test, y_test)
    
    conf_matrix = confusion_matrix(true_labels.flatten(), predictions.flatten())
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
    sample_problem = encode_labels(np.array(['23+17']))
    predicted_solution = text2text_model.predict(sample_problem)
    decoded_solution = decode_labels(predicted_solution[0])
    print(f"Predicted Solution: {decoded_solution}")

    return test_acc

if __name__ == "__main__":
    epochs=200
    learning_rate=1.e-3
    dropout_rate=0.02
    unique_characters = '0123456789+- '
    
    test_accuracies = []
    test_sizes = np.arange(0.1, 1.0, 0.2)  # Creates an array from 0.1 to 0.9 with a step of 0.2
    for test_size in test_sizes:
        print(f"test size {test_size:2.1f}")
        test_accuracy = main(learning_rate, dropout_rate, epochs, test_size)
        test_accuracies.append(test_accuracy)

    # Assuming test_sizes and test_accuracies are defined and contain your data
    np.savez('./results/text2text_53_test_performance.npz', test_sizes=test_sizes, test_accuracies=test_accuracies)
    print("Data saved successfully.")

    plt.plot(test_sizes, test_accuracies)
    plt.scatter(test_sizes, test_accuracies, marker='*', s=50)

    # Loop through each point to place a text label
    for i, txt in enumerate(test_accuracies):
        plt.text(test_sizes[i], test_accuracies[i] - 0.01, f'{txt:.2f}', ha='center', va='top')  

    plt.xlabel('test_size t')
    plt.ylabel('Test Accuracy')
    plt.ylim([0, 1])
    plt.title('text2text_53 Model Performance Across Test Sizes')
    plt.tight_layout()
    plt.savefig('./figures/experiment_1: text2text_53 accuracy across test sizes.pdf')
    # plt.show()
            
