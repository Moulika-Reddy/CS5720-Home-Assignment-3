
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------
# Home Assignment 3 - Question 2
# Transfer Learning: Freeze vs Fine-Tune
# -----------------------------------------

# Set a random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Step 1: Load the CIFAR-10 dataset
(x_train_all, y_train_all), (x_test_all, y_test_all) = (
    tf.keras.datasets.cifar10.load_data()
)

# Select only cats (3) and dogs (5)
train_mask = np.isin(y_train_all.flatten(), [3, 5])
test_mask = np.isin(y_test_all.flatten(), [3, 5])

x_train_all = x_train_all[train_mask]
y_train_all = y_train_all.flatten()[train_mask]

x_test_all = x_test_all[test_mask]
y_test_all = y_test_all.flatten()[test_mask]

# Convert labels: cat = 0, dog = 1
y_train_all = (y_train_all == 5).astype("float32")
y_test_all = (y_test_all == 5).astype("float32")

# Step 2: Create balanced training and validation sets
# Use 1000 cats and 1000 dogs for training
# Use 250 cats and 250 dogs for validation
rng = np.random.default_rng(42)

cat_indices = rng.permutation(np.where(y_train_all == 0)[0])
dog_indices = rng.permutation(np.where(y_train_all == 1)[0])

train_indices = np.concatenate([
    cat_indices[:1000],
    dog_indices[:1000]
])

val_indices = np.concatenate([
    cat_indices[1000:1250],
    dog_indices[1000:1250]
])

rng.shuffle(train_indices)
rng.shuffle(val_indices)

x_train = x_train_all[train_indices]
y_train = y_train_all[train_indices]

x_val = x_train_all[val_indices]
y_val = y_train_all[val_indices]

# Select 250 cats and 250 dogs for testing
test_cat_indices = rng.permutation(
    np.where(y_test_all == 0)[0]
)[:250]

test_dog_indices = rng.permutation(
    np.where(y_test_all == 1)[0]
)[:250]

test_indices = np.concatenate([
    test_cat_indices,
    test_dog_indices
])

rng.shuffle(test_indices)

x_test = x_test_all[test_indices]
y_test = y_test_all[test_indices]

print("Training images:", len(x_train))
print("Validation images:", len(x_val))
print("Testing images:", len(x_test))

# Step 3: Preprocess images for MobileNetV2
def preprocess(image, label):
    image = tf.image.resize(image, (96, 96))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def make_dataset(images, labels, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels)
    )

    dataset = dataset.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(images),
            seed=42,
            reshuffle_each_iteration=False
        )

    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_dataset = make_dataset(x_train, y_train, shuffle=True)
val_dataset = make_dataset(x_val, y_val)
test_dataset = make_dataset(x_test, y_test)

# Step 4: Create a MobileNetV2 classification model
def create_model(fine_tune=False):
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(96, 96, 3)
    )

    if fine_tune:
        # Unfreeze the last convolutional block
        base_model.trainable = True

        for layer in base_model.layers:
            layer.trainable = (
                layer.name.startswith("block_16")
                or layer.name.startswith("Conv_1")
            )

        # Keep Batch Normalization layers frozen
        for layer in base_model.layers:
            if isinstance(
                layer, tf.keras.layers.BatchNormalization
            ):
                layer.trainable = False
    else:
        # Freeze the entire pretrained network
        base_model.trainable = False

    inputs = tf.keras.Input(shape=(96, 96, 3))

    # Keep Batch Normalization statistics fixed
    x = base_model(inputs, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    learning_rate = 0.0001 if fine_tune else 0.001

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Step 5: Train and evaluate a model
def run_experiment(model, experiment_name):
    trainable_params = int(np.sum([
        np.prod(weight.shape)
        for weight in model.trainable_weights
    ]))

    print("\n" + experiment_name)
    print("Trainable Parameters:", trainable_params)

    start_time = time.time()

    history = model.fit(
        train_dataset,
        epochs=3,
        validation_data=val_dataset,
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate only after training is finished
    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=0
    )

    print("\nResults:", experiment_name)
    print("Trainable Parameters:", trainable_params)
    print("Training Time:", round(training_time, 2), "seconds")
    print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
    print("Training Loss:", history.history["loss"])

    return {
        "parameters": trainable_params,
        "time": training_time,
        "accuracy": test_accuracy,
        "history": history
    }

# Step 6: Experiment A - Frozen Feature Extraction
tf.keras.utils.set_random_seed(42)
model_a = create_model(fine_tune=False)

# Save the initial classifier weights so that
# Experiment B starts with the same classifier
classifier_weights = model_a.layers[-1].get_weights()

results_a = run_experiment(
    model_a,
    "Experiment A: Frozen Feature Extraction"
)

# Step 7: Experiment B - Fine-Tuning
tf.keras.utils.set_random_seed(42)
model_b = create_model(fine_tune=True)

# Use the same initial classifier weights
model_b.layers[-1].set_weights(classifier_weights)

results_b = run_experiment(
    model_b,
    "Experiment B: Fine-Tuning"
)

# Step 8: Print comparison table
print("\nComparison of Transfer Learning Methods")

print(
    f"{'Method':<28}"
    f"{'Parameters':<15}"
    f"{'Time (s)':<12}"
    f"{'Test Accuracy':<15}"
)

print("-" * 70)

print(
    f"{'Frozen Feature Extractor':<28}"
    f"{results_a['parameters']:<15}"
    f"{results_a['time']:<12.2f}"
    f"{results_a['accuracy'] * 100:.2f}%"
)

print(
    f"{'Fine-Tuned Network':<28}"
    f"{results_b['parameters']:<15}"
    f"{results_b['time']:<12.2f}"
    f"{results_b['accuracy'] * 100:.2f}%"
)

# Step 9: Plot training loss
epochs = range(1, 4)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    results_a["history"].history["loss"],
    marker="o",
    label="Frozen Feature Extractor"
)

plt.plot(
    epochs,
    results_b["history"].history["loss"],
    marker="s",
    label="Fine-Tuned Network"
)

plt.title("Training Loss: Frozen vs Fine-Tuned MobileNetV2")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True)

plt.savefig(
    "task2_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nTraining loss graph saved as task2_training_loss.png")
