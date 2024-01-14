import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.interpolate import interp1d

def plotconfusion(model,test_generator):
  # Make predictions on the test set
  y_pred = model.predict(test_generator)

  # Get true labels
  y_true = test_generator.classes

  # Convert predictions to class labels
  predicted_classes = np.argmax(y_pred, axis=1)

  # Calculate the confusion matrix
  cm = confusion_matrix(y_true, predicted_classes)

  # Visualize the confusion matrix
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Class 0", "Class 1", "Class 2"])
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.title("Confusion Matrix")
  plt.show()


def lr_schedule(epoch):
    """Learning Rate Schedule"""
    learning_rate = 1e-3
    if epoch > 2:
        learning_rate *= 0.1
    if epoch > 4:
        learning_rate *= 0.1
    return learning_rate


def get_example_predictions(num_samples, model,generator,testpath, sigmoid=False):
  # Get a random sample of test images and their true labels
  sample_indices = random.sample(range(len(generator.filenames)), num_samples)
  sample_images = [generator.filenames[i] for i in sample_indices]
  sample_labels = [generator.classes[i] for i in sample_indices]
  class_names = ['Class 0', 'Class 1', 'Class 2']
  # Make predictions on the sample images
  sample_predictions = model.predict(generator)

  # Convert predictions to class labels
  if sigmoid:
    sample_predicted_classes = (sample_predictions > 0.5).astype(int)
  else:
    sample_predicted_classes = np.argmax(sample_predictions, axis=1)

  # Display sample images with their true and predicted labels
  for i in range(num_samples):
      img_path = os.path.join(testpath, sample_images[i])
      img = plt.imread(img_path)
      plt.imshow(img)
      plt.title(f"True: {class_names[sample_labels[i]]}, Predicted: {class_names[sample_predicted_classes[i]]}")
      plt.show()


def plotroccurve(model, test_generator, num_thresholds = 1000):
    # Get true labels and predicted probabilities
    # Get true labels and predicted probabilities
    y_true = test_generator.classes
    y_score = model.predict(test_generator)

    # Binarize the true labels for each class
    y_bin = label_binarize(y_true, classes=np.unique(y_true))

    # Initialize lists to store ROC-AUC values for each class
    roc_auc_list = []

    # Plot ROC curves for each class
    plt.figure(figsize=(10, 8))

    for class_idx in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_score[:, class_idx])
        roc_auc = auc(fpr, tpr)
        roc_auc_list.append(roc_auc)

        # Plot ROC curve for each class
        plt.plot(fpr, tpr, label=f'Class {class_idx} (AUC = {roc_auc:.2f})')

    # Compute micro-average ROC-AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Plot micro-average ROC curve
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-Average (AUC = {roc_auc_micro:.2f})', linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Test Set')
    plt.legend()
    plt.show()

    # Print micro-average AUC
    print(f'Micro-Average AUC: {roc_auc_micro:.2f}')