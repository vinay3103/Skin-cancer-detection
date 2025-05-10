import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import os
import shutil
import random

# --- PARAMETERS ---
original_data_path = 'D:/Coding/Skin cancer ISIC The International Skin Imaging Collaboration/Augmented'
base_output_path = 'D:/Coding/Skin cancer ISIC The International Skin Imaging Collaboration/PreparedSplit'
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
# --- 1. Stratified Split ---
def prepare_stratified_split(original_dir, output_dir, train_split=0.7, val_split=0.15, test_split=0.15):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for cls in os.listdir(original_dir):
        cls_path = os.path.join(original_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = os.listdir(cls_path)
        random.shuffle(images)

        train_end = int(train_split * len(images))
        val_end = int((train_split + val_split) * len(images))

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split in splits:
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in splits[split]:
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copyfile(src, dst)

# Run split
prepare_stratified_split(original_data_path, base_output_path)

# --- 2. Load Datasets with Augmentation ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(base_output_path, 'train'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(base_output_path, 'val'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(base_output_path, 'test'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Class Info
class_names = train_ds.class_names
num_classes = len(class_names)

# Class weights
y_train_labels = np.concatenate([y.numpy() for _, y in train_ds])
y_train_int = np.argmax(y_train_labels, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
class_weight_dict = dict(enumerate(class_weights))

# Prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Model Architecture ---
base_model = MobileNetV3Small(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Phase 1: frozen

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Callbacks
checkpoint_cb = ModelCheckpoint("skin_cancer_BESTmodel.keras", save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# --- 4. Phase 1: Train head only ---
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_cb]
)

# --- 5. Phase 2: Fine-tune entire model ---
base_model.trainable = True
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

history= model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=136,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_cb]
)



# Predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# --- 7. Plotting ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.show()

# Save final model
model.save("skin_cancer_model.keras")
