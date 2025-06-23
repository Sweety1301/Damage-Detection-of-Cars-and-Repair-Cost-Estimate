import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

img_size = (224, 224) 
class_count = 3 

base_model = VGG19(weights='/kaggle/input/vgg19-weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(img_size[0], img_size[1], 3))

x = base_model.output
x = Flatten()(x) 
x = Dense(256, activation='relu')(x)  
output = Dense(class_count, activation='softmax')(x)  
model_vgg = Model(inputs=base_model.input, outputs=output)

model_vgg.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

epochs=50

history_vgg = model_vgg.fit(
    train_ds,
    epochs=epochs,
    validation_data=valid_ds,
    verbose=1,
)
all_train_histories = [history_vgg.history['accuracy']]
all_val_histories = [history_vgg.history['val_accuracy']]

model_vgg.save('model_vgg19.h5')

plt.figure(figsize=(12, 6))
plt.plot(history_vgg.history['accuracy'], label='Training Accuracy')
plt.plot(history_vgg.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

from sklearn.metrics import classification_report
true_labels = test_ds.classes
predictions = model_vgg.predict(test_ds)
predicted_labels = np.argmax(predictions, axis=1)
print(classification_report(true_labels, predicted_labels))
