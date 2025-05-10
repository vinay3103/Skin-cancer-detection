import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# Constants
IMG_SIZE = 224
NUM_CLASSES = 9
BATCH_SIZE = 32
MODEL_PATH = r'D:/vinay/programming codeas apps/SkinCancerDetection/skincancer/main/skin_cancer_model.keras'
TEST_DATA_PATH = "D:/Coding/Skin cancer ISIC The International Skin Imaging Collaboration/Augmented4"

# Load and preprocess test dataset
print("🔍 Loading test dataset...")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DATA_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True

)

# Rebuild the model architecture
print("🧠 Rebuilding model architecture...")
base_model = MobileNetV3Small(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = True

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=True)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load the trained weights
print(f"📦 Loading weights from '{MODEL_PATH}'...")
model.load_weights(MODEL_PATH)

# Evaluate the model
print("✅ Evaluating on test data...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\n🎯 Final Test Accuracy: {test_accuracy * 100:.2f}%")
