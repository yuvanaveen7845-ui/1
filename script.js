
const programs = [
    {
        id: 1,
        name: "Supervised Learning Classification",
        language: "Python",
        code: `from sklearn.datasets import load_iris
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
print("Confusion Matrix:\\n", confusion_matrix(y_test, predictions))`
    },
    {
        id: 2,
        name: "Build a Deep Learning Model",
        language: "Python",
        code: `import tensorflow as tf
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
print("Accuracy:", accuracy_score(y_test, predictions))`
    },
    {
        id: 3,
        name: "Forward Propagation Using NumPy",
        language: "Python",
        code: `import numpy as np

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
print("Network Output:", output)`
    },
    {
        id: 4,
        name: "Image Classification Using CNN",
        language: "Python",
        code: `import tensorflow as tf

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
print("Test Results:", model.evaluate(X_test, y_test))`
    },
    {
        id: 5,
        name: "Text Classification Using CNN",
        language: "Python",
        code: `import tensorflow as tf

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
print("Test Results:", model.evaluate(X_test, y_test))`
    },
    {
        id: 6,
        name: "Traffic Object Classification Using CNN",
        language: "Python",
        code: `import tensorflow as tf

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
# and a detector such as YOLO or Faster R-CNN.`
    },
    {
        id: 7,
        name: "Implementation of RNN",
        language: "Python",
        code: `import numpy as np
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
print(sample)`
    },
    {
        id: 8,
        name: "Image Augmentation Using Deep RBM",
        language: "Python",
        code: `import numpy as np
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
# The RBM learns features but does not generate the augmented image.`
    },
    {
        id: 9,
        name: "Sentiment Analysis Using LSTM",
        language: "Python",
        code: `import tensorflow as tf

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

# IMDB labels: 0 = negative, 1 = positive`
    },
    {
        id: 10,
        name: "Variational Autoencoders",
        language: "Python",
        code: `import tensorflow as tf
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

print("VAE Training Completed")`
    }
];

const programList = document.getElementById("programList");
const search = document.getElementById("search");
const title = document.getElementById("programTitle");
const description = document.getElementById("programDescription");
const language = document.getElementById("codeLanguage");
const codeDisplay = document.getElementById("codeDisplay");
const copyBtn = document.getElementById("copyBtn");
const copyStatus = document.getElementById("copyStatus");

let selectedProgram = programs[0];

function renderPrograms(filter = "") {
    programList.replaceChildren();

    const filtered = programs.filter(program =>
        (`program ${program.id} ${program.name} ${program.language}`)
            .toLowerCase()
            .includes(filter.toLowerCase())
    );

    if (filtered.length === 0) {
        const message = document.createElement("p");
        message.textContent = "No programs found.";
        message.className = "hint";
        programList.appendChild(message);
        return;
    }

    filtered.forEach(program => {
        const button = document.createElement("button");
        button.className = "program-btn";
        button.textContent = `Program ${program.id} - ${program.name}`;

        if (selectedProgram.id === program.id) {
            button.classList.add("active");
        }

        button.addEventListener("click", () => showProgram(program));
        programList.appendChild(button);
    });
}

function showProgram(program) {
    selectedProgram = program;
    title.textContent = `Program ${program.id}`;
    description.textContent = program.name;
    language.textContent = program.language.toUpperCase() + " SOURCE CODE";
    codeDisplay.textContent = program.code;
    copyStatus.textContent = "Click Copy Code to copy the full program.";
    copyBtn.textContent = "📋 Copy Code";
    renderPrograms(search.value);
}

async function copyCode() {
    try {
        await navigator.clipboard.writeText(selectedProgram.code);
        copyBtn.textContent = "✓ Copied!";
        copyStatus.textContent = "Code copied successfully!";

        setTimeout(() => {
            copyBtn.textContent = "📋 Copy Code";
        }, 1500);
    } catch (error) {
        const textarea = document.createElement("textarea");
        textarea.value = selectedProgram.code;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();

        const copied = document.execCommand("copy");
        textarea.remove();

        copyStatus.textContent = copied
            ? "Code copied successfully!"
            : "Copy failed. Please copy manually.";

        copyBtn.textContent = copied ? "✓ Copied!" : "📋 Copy Code";
    }
}

search.addEventListener("input", () => {
    renderPrograms(search.value);
});

copyBtn.addEventListener("click", copyCode);

showProgram(programs[0]);
