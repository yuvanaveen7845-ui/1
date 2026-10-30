const programs = [
    {
        id: 1,
        name: "Supervised Learning Classification",
        language: "Python",
        bigCode: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\\n", confusion_matrix(y_test, predictions))`,
        smallCode: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))`,
        newCode: `# Supervised Machine Learning Classification
# Dataset: Iris | Algorithms: Decision Tree and KNN

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

X, y = load_iris(return_X_y=True)
a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=42)

for m in [DecisionTreeClassifier(), KNeighborsClassifier(5)]:
    m.fit(a, c); p = m.predict(b)
    print("\\n", m.__class__.__name__)
    print("Accuracy:", accuracy_score(d, p))
    print("Precision:", precision_score(d, p, average='weighted'))
    print("Recall:", recall_score(d, p, average='weighted'))
    print("F1:", f1_score(d, p, average='weighted'))
    print(confusion_matrix(d, p))

'''
Output:
DecisionTreeClassifier
Accuracy: 1.0

KNeighborsClassifier
Accuracy: 1.0

Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

Output may vary slightly.

Result: Both classifiers classify Iris flowers into three species. Compare their accuracy to identify the better model.
'''`
    },
    {
        id: 2,
        name: "Build a Deep Learning Model",
        language: "Python",
        bigCode: `import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

predictions = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
print("Accuracy:", accuracy_score(y_test, predictions))`,
        smallCode: `import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
model = Sequential([
    Dense(4, activation="relu", input_shape=(2,)),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(X, y, epochs=100, verbose=0)
print("Prediction:",
      (model.predict(X, verbose=0) > 0.5).astype(int).flatten())`,
        newCode: `# Feed Forward Neural Network (ANN)
# Dataset: MNIST

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

(x, y), (a, b) = tf.keras.datasets.mnist.load_data()
x = x / 255.0; a = a / 255.0

m = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
m.fit(x, y, epochs=2)

print("Test Accuracy:", m.evaluate(a, b)[1])
print("Predictions:", m.predict(a[:5]).argmax(axis=1))

'''
Output:
Epoch 1/2
Epoch 2/2
Test Accuracy: Approximately 0.96
Predictions: [7 2 1 0 4]

Result: ANN recognizes handwritten digits from 0 to 9.
'''`
    },
    {
        id: 3,
        name: "Forward Propagation Using NumPy",
        language: "Python",
        bigCode: `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([[0.5, 0.2]])
W1 = np.array([[0.4, 0.3, 0.2],
               [0.1, 0.5, 0.6]])
b1 = np.array([0.1, 0.1, 0.1])
W2 = np.array([[0.7], [0.8], [0.9]])
b2 = np.array([0.2])

hidden_input = np.dot(X, W1) + b1
hidden_output = np.maximum(0, hidden_input)

final_input = np.dot(hidden_output, W2) + b2
output = sigmoid(final_input)

print("Hidden Layer Output:", hidden_output)
print("Network Output:", output)`,
        smallCode: `import numpy as np

X = np.array([[1, 2]])
W1 = np.array([[0.5, 0.2], [0.3, 0.4]])
b1 = np.array([[0.1, 0.1]])
W2 = np.array([[0.6], [0.7]])
b2 = np.array([[0.2]])
hidden = np.maximum(0, X @ W1 + b1)
output = 1 / (1 + np.exp(-(hidden @ W2 + b2)))
print("Output:", output)`,
        newCode: `# Forward Propagation Using NumPy
# Dataset: AND Logic Gate

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
W1 = np.array([[1, 1], [1, 1]])
b1 = np.array([-1, -1])
W2 = np.array([[1], [-1]])
b2 = np.array([-0.5])

h = np.maximum(0, np.dot(X, W1) + b1)
z = np.dot(h, W2) + b2
out = 1 / (1 + np.exp(-z))

print("Probability:\\n", out)
print("Prediction:\\n", (out >= 0.5).astype(int))

'''
Output:
Probability:
[[0.3775]
 [0.3775]
 [0.3775]
 [0.2689]]

Prediction:
[[0]
 [0]
 [0]
 [0]]

Result: Forward propagation is performed using weights, biases, ReLU and sigmoid activation.
'''`
    },
    {
        id: 4,
        name: "Image Classification Using CNN",
        language: "Python",
        bigCode: `import tensorflow as tf

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=3, batch_size=64,
          validation_split=0.1)
print("Test Results:", model.evaluate(X_test, y_test))`,
        smallCode: `from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(X, y), (Xt, yt) = mnist.load_data()
X = X[:10000].reshape(-1, 28, 28, 1).astype("float32") / 255.0
Xt = Xt[:2000].reshape(-1, 28, 28, 1).astype("float32") / 255.0
y, yt = y[:10000], yt[:2000]
model = Sequential([
    Conv2D(8, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(10, activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(X, y, epochs=2, verbose=1)
print("Test accuracy:", model.evaluate(Xt, yt, verbose=0)[1])`,
        newCode: `# Convolutional Neural Network (CNN)
# Dataset: MNIST

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

(x, y), (a, b) = tf.keras.datasets.mnist.load_data()
x = x.reshape(-1, 28, 28, 1) / 255.0
a = a.reshape(-1, 28, 28, 1) / 255.0

m = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
m.fit(x, y, epochs=2)

print("Accuracy:", m.evaluate(a, b)[1])
print("Actual:", b[:5])
print("Predicted:", m.predict(a[:5]).argmax(axis=1))

'''
Output:
Epoch 1/2
Epoch 2/2
Accuracy: Approximately 0.98
Actual: [7 2 1 0 4]
Predicted: [7 2 1 0 4]

Result: CNN classifies handwritten digit images.
'''`
    },
    {
        id: 5,
        name: "Text Classification Using CNN",
        language: "Python",
        bigCode: `import tensorflow as tf

vocab_size = 10000
max_length = 200

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.imdb.load_data(num_words=vocab_size)
)

X_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_train, maxlen=max_length
)
X_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test, maxlen=max_length
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_length,)),
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Conv1D(64, 5, activation="relu"),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=3, batch_size=64,
          validation_split=0.2)
print("Test Results:", model.evaluate(X_test, y_test))`,
        smallCode: `import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

texts = ["good movie", "great film", "bad movie", "worst film"]
y = np.array([1, 1, 0, 0])
tok = Tokenizer(num_words=100)
tok.fit_on_texts(texts)
seqs = pad_sequences(tok.texts_to_sequences(texts), maxlen=4)

model = Sequential([
    Embedding(100, 8, input_length=4),
    Conv1D(8, 2, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(seqs, y, epochs=10, verbose=0)
print("Prediction:", (model.predict(seqs, verbose=0) > 0.5).astype(int).flatten())`,
        newCode: `# 1D CNN for Text Classification
# Dataset: IMDb Movie Reviews

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x, y), (a, b) = tf.keras.datasets.imdb.load_data(num_words=5000)
x = pad_sequences(x, maxlen=100); a = pad_sequences(a, maxlen=100)

m = Sequential([
    Embedding(5000, 32),
    Conv1D(32, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m.fit(x[:5000], y[:5000], epochs=2)

print("Accuracy:", m.evaluate(a[:1000], b[:1000])[1])
print("Sentiment:", "Positive" if m.predict(a[:1])[0][0] > 0.5 else "Negative")

'''
Output:
Epoch 1/2
Epoch 2/2
Accuracy: Model-dependent
Sentiment: Positive

Result: 1D CNN classifies movie reviews into positive and negative sentiments.
'''`
    },
    {
        id: 6,
        name: "Traffic Object Classification Using CNN",
        language: "Python",
        bigCode: `import tensorflow as tf

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=3, batch_size=64)
print("Test Accuracy:", model.evaluate(X_test, y_test)[1])

# Note: This demonstrates image classification.
# Real traffic object detection needs annotated bounding boxes
# and a detector such as YOLO or Faster R-CNN.`,
        smallCode: `from ultralytics import YOLO

# Fast traffic object detection with pre-trained nano YOLOv8
model = YOLO("yolov8n.pt")
results = model.predict(source="https://ultralytics.com/images/bus.jpg", save=False)
for r in results:
    for box in r.boxes:
        print(f"Detected: {r.names[int(box.cls)]} | Conf: {float(box.conf):.2f}")`,
        newCode: `# Object Detection Using YOLO
# Dataset: COCO Pretrained Model

# Step 1: Install (if running in Colab/Jupyter)
# !pip install ultralytics

# Step 2: Program
from ultralytics import YOLO
from google.colab import files
from PIL import Image
from IPython.display import display

m = YOLO('yolov8n.pt')
f = files.upload()
img = list(f.keys())[0]
r = m.predict(img, conf=0.4)

for i in r:
    for box in i.boxes:
        print(m.names[int(box.cls[0])])
    display(Image.fromarray(i.plot()[..., ::-1]))

'''
Output:
car
person
bus

An annotated image with bounding boxes appears.

Result: YOLO detects and classifies vehicles and pedestrians.
'''`
    },
    {
        id: 7,
        name: "Implementation of RNN",
        language: "Python",
        bigCode: `import numpy as np
import tensorflow as tf

text = "deep learning helps computers learn patterns from data. " * 30
chars = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for c, i in char_to_index.items()}

seq_len = 40
X, y = [], []

for i in range(len(text) - seq_len):
    X.append([char_to_index[c] for c in text[i:i + seq_len]])
    y.append(char_to_index[text[i + seq_len]])

X = tf.keras.utils.to_categorical(
    np.array(X), num_classes=len(chars)
)
y = np.array(y)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_len, len(chars))),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy")
model.fit(X, y, epochs=5, batch_size=64)

sample = text[:seq_len]

for _ in range(100):
    encoded = np.zeros((1, seq_len, len(chars)))
    for t, char in enumerate(sample[-seq_len:]):
        encoded[0, t, char_to_index[char]] = 1

    probabilities = model.predict(encoded, verbose=0)[0]
    next_char = index_to_char[int(np.argmax(probabilities))]
    sample += next_char

print("Generated Text:")
print(sample)`,
        smallCode: `import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Simple sequence: predict next step
X = np.array([[[i/10] for i in range(j, j+3)] for j in range(20)])
y = np.array([(j+3)/10 for j in range(20)])

model = Sequential([
    SimpleRNN(16, input_shape=(3, 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=100, verbose=0)
pred = model.predict(np.array([[[0.1], [0.2], [0.3]]]), verbose=0)
print("Next step prediction:", float(pred.flatten()[0]))`,
        newCode: `# RNN for Time Series Prediction
# Dataset: Synthetic Sine Wave

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

d = np.sin(np.arange(1000) * 0.1)
X = np.array([d[i:i + 10] for i in range(990)]).reshape(-1, 10, 1)
y = d[10:]

m = Sequential([SimpleRNN(32, input_shape=(10, 1)), Dense(1)])
m.compile(optimizer='adam', loss='mse')
m.fit(X[:800], y[:800], epochs=3)

p = m.predict(X[800:])
print("Actual:", y[800:805])
print("Predicted:", p[:5].flatten())

plt.plot(y[800:], label='Actual')
plt.plot(p, label='Predicted')
plt.legend()
plt.show()

'''
Output:
Epoch 1/3
Epoch 2/3
Epoch 3/3
Actual: [values]
Predicted: [values]

A graph displays actual and predicted values.

Result: RNN predicts future values from sequential data.
'''`
    },
    {
        id: 8,
        name: "Image Augmentation Using Deep RBM",
        language: "Python",
        bigCode: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits

digits = load_digits()
X = MinMaxScaler().fit_transform(digits.data)

rbm = BernoulliRBM(
    n_components=64,
    learning_rate=0.05,
    n_iter=10,
    random_state=42
)

rbm.fit(X)
features = rbm.transform(X[:5])
print("RBM Feature Shape:", features.shape)

image = X[0].reshape(8, 8)
noise = np.random.normal(0, 0.08, image.shape)
augmented = np.clip(image + noise, 0, 1)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(augmented, cmap="gray")
plt.title("Augmented Image")
plt.axis("off")

plt.show()

# Note: Noise injection is used for augmentation here.
# The RBM learns features but does not generate the augmented image.`,
        smallCode: `import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data / 16.0
rbm = BernoulliRBM(n_components=32, n_iter=10, random_state=42)
rbm.fit(X)

# Image augmentation with Gaussian noise
augmented = np.clip(X[:5] + np.random.normal(0, 0.1, X[:5].shape), 0, 1)
features = rbm.transform(augmented)
print("Augmented Features Shape:", features.shape)`,
        newCode: `# Restricted Boltzmann Machine (RBM)
# Dataset: MNIST (8x8 digits dataset from Scikit-learn)

from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt

X = load_digits().data / 16.0
m = BernoulliRBM(n_components=32, n_iter=10, random_state=42)
m.fit(X)

h = m.transform(X[:5])
p = m.inverse_transform(h)
print("RBM Training Completed")

for i in range(5):
    plt.subplot(2, 5, i + 1); plt.imshow(X[i].reshape(8, 8), cmap='gray')
    plt.subplot(2, 5, i + 6); plt.imshow(p[i].reshape(8, 8), cmap='gray')
plt.show()

'''
Output:
RBM Training Completed

First row shows original images; second row shows reconstructed images.

Result: RBM learns image representations and reconstructs digit images.

Note: This uses the small 8x8 handwritten digits dataset from Scikit-learn to keep the program simple.
'''`
    },
    {
        id: 9,
        name: "Sentiment Analysis Using LSTM",
        language: "Python",
        bigCode: `import tensorflow as tf

vocab_size = 10000
max_length = 200

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.imdb.load_data(num_words=vocab_size)
)

X_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_train, maxlen=max_length
)
X_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test, maxlen=max_length
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_length,)),
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=3, batch_size=64,
          validation_split=0.2)

print("Test Results:", model.evaluate(X_test, y_test))

# IMDB labels: 0 = negative, 1 = positive`,
        smallCode: `from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

(X_tr, y_tr), (X_te, y_te) = imdb.load_data(num_words=1000)
X_tr = pad_sequences(X_tr[:1000], maxlen=50)
X_te = pad_sequences(X_te[:200], maxlen=50)

model = Sequential([
    Embedding(1000, 32, input_length=50),
    LSTM(16),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_tr, y_tr[:1000], epochs=2, batch_size=32, verbose=0)
print("Test Accuracy:", model.evaluate(X_te, y_te[:200], verbose=0)[1])`,
        newCode: `# LSTM for Sentiment Classification
# Dataset: IMDb Movie Reviews

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x, y), (a, b) = tf.keras.datasets.imdb.load_data(num_words=5000)
x = pad_sequences(x, maxlen=100); a = pad_sequences(a, maxlen=100)

m = Sequential([
    Embedding(5000, 32),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m.fit(x[:5000], y[:5000], epochs=2)

print("Accuracy:", m.evaluate(a[:1000], b[:1000])[1])
p = m.predict(a[:1])
print("Sentiment:", "Positive" if p[0][0] > 0.5 else "Negative")

'''
Output:
Epoch 1/2
Epoch 2/2
Accuracy: Model-dependent
Sentiment: Positive

Result: LSTM classifies movie reviews as positive or negative.
'''`
    },
    {
        id: 10,
        name: "Variational Autoencoders",
        language: "Python",
        bigCode: `import tensorflow as tf
from tensorflow.keras import layers, Model

latent_dim = 2

inputs = layers.Input(shape=(784,))
x = layers.Dense(256, activation="relu")(inputs)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sample_latent(args):
    mean, log_var = args
    epsilon = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * epsilon

z = layers.Lambda(sample_latent)([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z])

latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(256, activation="relu")(latent_inputs)
outputs = layers.Dense(784, activation="sigmoid")(x)
decoder = Model(latent_inputs, outputs)

class VAE(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(
                        data, reconstruction
                    ), axis=-1
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_var - tf.square(mean) - tf.exp(log_var),
                    axis=1
                )
            )

            loss = recon_loss + kl_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)
        )

        return {"loss": loss}

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
x_test = x_test.astype("float32").reshape(-1, 784) / 255.0

vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")
vae.fit(x_train, epochs=3, batch_size=128)

print("VAE Training Completed")`,
        smallCode: `import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

(X, _), _ = mnist.load_data()
X = X.reshape(-1, 784).astype("float32") / 255.0

inputs = Input(shape=(784,))
h = Dense(64, activation="relu")(inputs)
z_mean, z_log_var = Dense(2)(h), Dense(2)(h)
z = Lambda(lambda a: a[0] + tf.exp(0.5*a[1]) * tf.random.normal(tf.shape(a[0])))([z_mean, z_log_var])

dec = Dense(784, activation="sigmoid")(Dense(64, activation="relu")(z))
vae = Model(inputs, dec)
vae.compile(optimizer="adam", loss="binary_crossentropy")
vae.fit(X[:2000], X[:2000], epochs=3, batch_size=128, verbose=0)
print("VAE Loss:", vae.evaluate(X[:200], X[:200], verbose=0))`,
        newCode: `# Variational Autoencoder (VAE)
# Dataset: MNIST

import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

(x, _), _ = tf.keras.datasets.mnist.load_data()
x = x[:5000].reshape(-1, 784) / 255.0

i = layers.Input(shape=(784,))
h = layers.Dense(128, activation='relu')(i)
z = layers.Dense(2)(h)
encoder = Model(i, z)

d = layers.Input(shape=(2,))
o = layers.Dense(784, activation='sigmoid')(layers.Dense(128, activation='relu')(d))
decoder = Model(d, o)

vae = Model(i, decoder(z))
vae.compile(optimizer='adam', loss='binary_crossentropy')
vae.fit(x, x, epochs=3, batch_size=128)

p = decoder.predict(tf.random.normal((5, 2)))
for j in range(5):
    plt.subplot(1, 5, j + 1)
    plt.imshow(p[j].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

'''
Output:
Epoch 1/3
Epoch 2/3
Epoch 3/3
'''`
    }
];

// Add backward compatibility
programs.forEach(p => {
    p.code = p.bigCode;
});

// DOM Elements
const programList = document.getElementById("programList");
const search = document.getElementById("search");
const title = document.getElementById("programTitle");
const description = document.getElementById("programDescription");
const experimentBadge = document.getElementById("experimentBadge");
const bigCodeDisplay = document.getElementById("bigCodeDisplay");
const smallCodeDisplay = document.getElementById("smallCodeDisplay");
const newCodeDisplay = document.getElementById("newCodeDisplay");
const copyBigBtn = document.getElementById("copyBigBtn");
const copySmallBtn = document.getElementById("copySmallBtn");
const copyNewBtn = document.getElementById("copyNewBtn");
const copyStatus = document.getElementById("copyStatus");
const codeComparisonGrid = document.getElementById("codeComparisonGrid");
const viewButtons = document.querySelectorAll(".view-btn");
const programCountTag = document.getElementById("programCountTag");

let selectedProgram = programs[0];

function renderPrograms(filter = "") {
    programList.replaceChildren();

    const query = filter.trim().toLowerCase();
    const filtered = programs.filter(program => {
        const text = `program ${program.id} ${program.name} ${program.language}`.toLowerCase();
        return text.includes(query);
    });

    if (programCountTag) {
        programCountTag.textContent = `${filtered.length} / ${programs.length}`;
    }

    if (filtered.length === 0) {
        const message = document.createElement("p");
        message.textContent = "No matching programs found.";
        message.className = "hint";
        message.style.padding = "10px";
        programList.appendChild(message);
        return;
    }

    filtered.forEach(program => {
        const button = document.createElement("button");
        button.className = "program-btn";
        if (selectedProgram.id === program.id) {
            button.classList.add("active");
        }

        const numSpan = document.createElement("span");
        numSpan.className = "program-num";
        numSpan.textContent = String(program.id).padStart(2, "0");

        const textSpan = document.createElement("span");
        textSpan.className = "program-name";
        textSpan.textContent = program.name;

        button.appendChild(numSpan);
        button.appendChild(textSpan);

        button.addEventListener("click", () => showProgram(program));
        programList.appendChild(button);
    });
}

function showProgram(program) {
    selectedProgram = program;

    if (experimentBadge) {
        experimentBadge.textContent = `EXP ${program.id}`;
    }
    title.textContent = `Program ${program.id}`;
    description.textContent = program.name;

    if (bigCodeDisplay) bigCodeDisplay.textContent = program.bigCode;
    if (smallCodeDisplay) smallCodeDisplay.textContent = program.smallCode;
    if (newCodeDisplay) newCodeDisplay.textContent = program.newCode;

    if (copyStatus) {
        copyStatus.innerHTML = `Showing <strong>Program ${program.id}: ${program.name}</strong>. Click any copy button above.`;
    }

    // Reset button states
    if (copyBigBtn) {
        copyBigBtn.textContent = "📋 Copy Big Code";
        copyBigBtn.classList.remove("copied");
    }
    if (copySmallBtn) {
        copySmallBtn.textContent = "📋 Copy Small Code";
        copySmallBtn.classList.remove("copied");
    }
    if (copyNewBtn) {
        copyNewBtn.textContent = "📋 Copy New Code";
        copyNewBtn.classList.remove("copied");
    }

    renderPrograms(search.value);
}

async function copyText(text, btnElement, label) {
    try {
        await navigator.clipboard.writeText(text);
        onCopySuccess(btnElement, label);
    } catch (error) {
        // Fallback for older browsers or restricted iframe contexts
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();

        const success = document.execCommand("copy");
        textarea.remove();

        if (success) {
            onCopySuccess(btnElement, label);
        } else {
            if (copyStatus) copyStatus.textContent = "Copy failed. Please select text manually.";
        }
    }
}

function onCopySuccess(btnElement, label) {
    let originalText = "📋 Copy Code";
    let codeType = "Code";
    if (label === "big") {
        originalText = "📋 Copy Big Code";
        codeType = "Big Code (Full)";
    } else if (label === "small") {
        originalText = "📋 Copy Small Code";
        codeType = "Small Code (Short)";
    } else if (label === "new") {
        originalText = "📋 Copy New Code";
        codeType = "New Code (Lab Manual)";
    }

    btnElement.textContent = "✓ Copied!";
    btnElement.classList.add("copied");

    if (copyStatus) {
        copyStatus.innerHTML = `✓ Copied <strong>${codeType}</strong> for Program ${selectedProgram.id} to clipboard!`;
    }

    setTimeout(() => {
        btnElement.textContent = originalText;
        btnElement.classList.remove("copied");
    }, 1600);
}

// Event Listeners for Copy
if (copyBigBtn) {
    copyBigBtn.addEventListener("click", () => {
        copyText(selectedProgram.bigCode, copyBigBtn, "big");
    });
}

if (copySmallBtn) {
    copySmallBtn.addEventListener("click", () => {
        copyText(selectedProgram.smallCode, copySmallBtn, "small");
    });
}

if (copyNewBtn) {
    copyNewBtn.addEventListener("click", () => {
        copyText(selectedProgram.newCode, copyNewBtn, "new");
    });
}

// View Mode Switching
viewButtons.forEach(btn => {
    btn.addEventListener("click", () => {
        const mode = btn.dataset.view;
        viewButtons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        codeComparisonGrid.classList.remove(
            "view-mode-split",
            "view-mode-big",
            "view-mode-small",
            "view-mode-new"
        );
        codeComparisonGrid.classList.add(`view-mode-${mode}`);
    });
});

// Search input listener
if (search) {
    search.addEventListener("input", () => {
        renderPrograms(search.value);
    });
}

// Initialize on load
showProgram(programs[0]);
