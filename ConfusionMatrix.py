import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
true_labels = test_ds.classes
predictions = model.predict(test_ds)
predicted_labels = np.argmax(predictions, axis=1)


plt.figure(figsize=(8, 10))
sns.heatmap(confusion_matrix(true_labels, predicted_labels), annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual Classes')
plt.show()
