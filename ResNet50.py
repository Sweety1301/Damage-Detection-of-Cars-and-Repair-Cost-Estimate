base_model = ResNet50(weights='/kaggle/input/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))


for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs=50

history_resnet = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=valid_ds,
    verbose=1,
)
all_train_histories = [history_resnet.history['accuracy']]
all_val_histories = [history_resnet.history['val_accuracy']]

model.save('model_rs50.h5')

plt.figure(figsize=(12, 6))
plt.plot(history_resnet.history['accuracy'], label='Training Accuracy')
plt.plot(history_resnet.history['val_accuracy'], label='Validation Accuracy')
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
