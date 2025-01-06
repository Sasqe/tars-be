# import keras.api.saving
# from keras._tf_keras.keras.models import Sequential
# from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# from keras._tf_keras.keras.datasets import mnist
# from keras._tf_keras.keras.optimizers import Adam
# from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
# from keras.api import callbacks
# import numpy as np
# import tensorflow as tf
#
# # Enable GPU memory growth (if applicable)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)
#
# # Load and preprocess the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # Normalize the images to the range [0, 1]
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0
#
# # Expand dimensions to include channel axis (grayscale images)
# x_train = np.expand_dims(x_train, axis=-1)
# x_test = np.expand_dims(x_test, axis=-1)
#
# # Convert labels to one-hot encoding
# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)
#
# # Data augmentation
# train_datagen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
# )
#
# train_generator = train_datagen.flow(x_train, y_train, batch_size=64)
#
# # Build the model
# model = Sequential([
#     Conv2D(32, (3, 3), strides=1, padding="same", activation='relu', input_shape=(28, 28, 1)),
#     BatchNormalization(),
#     Conv2D(64, (3, 3), strides=2, padding="same", activation='relu'),
#     BatchNormalization(),
#     Dropout(0.25),
#
#     Conv2D(128, (3, 3), strides=1, padding="same", activation='relu'),
#     BatchNormalization(),
#     Conv2D(128, (3, 3), strides=2, padding="same", activation='relu'),
#     BatchNormalization(),
#     Dropout(0.25),
#
#     Flatten(),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(10, activation='softmax')
# ])
#
# # Compile the model
# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# # Define callbacks
# checkpoint_cb = callbacks.ModelCheckpoint(
#     'best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max'
# )
# early_stopping_cb = callbacks.EarlyStopping(
#     monitor='val_accuracy', patience=10, restore_best_weights=True
# )
# reduce_lr_cb = callbacks.ReduceLROnPlateau(
#     monitor='val_loss', factor=0.5, patience=5, verbose=1
# )
#
# # Train the model
# history = model.fit(
#     train_generator,
#     epochs=50,
#     validation_data=(x_test, y_test),
#     callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
#     steps_per_epoch=len(x_train) // 64
# )
#
# # Evaluate the model on the test set
# final_loss, final_accuracy = model.evaluate(x_test, y_test, verbose=0)
# print(f"Test Accuracy: {final_accuracy * 100:.2f}%")
#
# # Save the model
# keras.api.saving.save_model(model, 'mnist_final_model.keras')
#
# # Optionally, plot training history
# import matplotlib.pyplot as plt
#
# def plot_history(history):
#     plt.figure(figsize=(12, 4))
#
#     # Plot accuracy
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.title('Accuracy Over Epochs')
#
#     # Plot loss
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Loss Over Epochs')
#
#     plt.show()
#
# plot_history(history)