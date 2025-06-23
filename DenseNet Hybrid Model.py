import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

inputs = layers.Input(shape=(224, 224, 3))

# Load ResNet50 with predefined weights
resnet_base = tf.keras.applications.ResNet50(
    include_top=False,
    weights='/kaggle/input/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    input_tensor=inputs
)
x_resnet = resnet_base(inputs)
x_resnet = layers.GlobalAveragePooling2D()(x_resnet)


densenet_base = tf.keras.applications.DenseNet169(
    include_top=False,
    weights='/kaggle/input/densenet/densenet169_weights_tf_dim_ordering_tf_kernels_notop (1).h5',
    input_shape=(224, 224, 3)
)
x_densenet = densenet_base(inputs)
x_densenet = layers.GlobalAveragePooling2D()(x_densenet)

x = layers.Concatenate()([x_resnet, x_densenet])

x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
x = layers.Dropout(rate=0.4)(x)
outputs = layers.Dense(3, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

epochs=50
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=valid_ds,  
    verbose=1,
)
all_train_histories = [history.history['accuracy']]
all_val_histories = [history.history['val_accuracy']]

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

from sklearn.metrics import classification_report
true_labels = test_ds.classes
predictions = model.predict(test_ds)
predicted_labels = np.argmax(predictions, axis=1)
print(classification_report(true_labels, predicted_labels))
