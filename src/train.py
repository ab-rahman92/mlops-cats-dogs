import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────
# Configuration - easy to change / log
# ────────────────────────────────────────────────

IMG_SIZE = (224, 224)
BATCH_SIZE = 2
EPOCHS = 64               # small for quick experiments; increase later
LEARNING_RATE = 0.001
AUGMENTATION = True

DATA_DIR = "data/processed"

# ────────────────────────────────────────────────
def create_simple_cnn(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')   # binary → cat vs dog
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ────────────────────────────────────────────────
def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20          if AUGMENTATION else 0,
        width_shift_range=0.15     if AUGMENTATION else 0,
        height_shift_range=0.15    if AUGMENTATION else 0,
        shear_range=0.1            if AUGMENTATION else 0,
        zoom_range=0.15            if AUGMENTATION else 0,
        horizontal_flip=True       if AUGMENTATION else False,
        fill_mode='nearest'
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    val_gen = valid_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_gen = valid_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print("Class indices:", train_gen.class_indices)  # should be {'cat': 0, 'dog': 1} or reverse
    
    return train_gen, val_gen, test_gen

# ────────────────────────────────────────────────
def plot_history(history, save_path="training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history.history['accuracy'], label='train acc')
    ax1.plot(history.history['val_accuracy'], label='val acc')
    ax1.set_title('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='train loss')
    ax2.plot(history.history['val_loss'], label='val loss')
    ax2.set_title('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

# ────────────────────────────────────────────────
def main():
    # ── MLflow setup ─────────────────────────────────
    mlflow.set_experiment("cats-vs-dogs-baseline")
    
    with mlflow.start_run(run_name="simple-cnn-run-1"):
        # Log hyperparameters
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "img_size": IMG_SIZE,
            "augmentation": AUGMENTATION,
            "optimizer": "Adam",
            "architecture": "Conv32-64-128 + Dense256-Drop0.5"
        })
        
        # Data
        train_gen, val_gen, test_gen = get_data_generators()
        
        # Model
        model = create_simple_cnn()
        model.summary(print_fn=lambda x: mlflow.log_text(x, "model_summary.txt"))
        
        # Train
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
        
        # Log final metrics
        mlflow.log_metrics({
            "train_acc_final": history.history['accuracy'][-1],
            "val_acc_final":   history.history['val_accuracy'][-1],
            "test_accuracy":   test_acc,
            "test_loss":       test_loss
        })
        
        # Log learning curves plot
        plot_path = plot_history(history)
        mlflow.log_artifact(plot_path, "plots/training_curves.png")
        
        # Save model (Keras format + MLflow format)
        model.save("models/baseline_model.keras")
        mlflow.keras.log_model(model, "model")
        
        print("Training finished. Model and artifacts logged to MLflow.")

if __name__ == "__main__":
    # Make sure output folder exists
    os.makedirs("models", exist_ok=True)
    main()