# Standard libraries
import os
import random
import shutil
from collections import Counter, defaultdict

# Third-party libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# TensorFlow and Keras
import keras
import tensorflow as tf
from keras import ops
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam



def display_batch(images, labels, num_images=64):
    plt.figure(figsize=(12, 12))
    grid_size = int(num_images**0.5) 
    
    for i in range(num_images):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i])  
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def organise_images_by_class(image_directory, parent_directory):
    """
    Organizes images into subdirectories based on their class names derived from the filenames.

    Parameters:
    - image_directory (str): Path to the directory containing the images.
    - parent_directory (str): Path to the parent directory where class subdirectories will be created.
    """
    classes = set()
    for image in os.listdir(image_directory):
        classes.add(image[:2])

    newlist = []
    for classname in list(classes):
        newstring = ""
        for char in classname:
            if char.isnumeric():
                newstring += char
        newlist.append(int(newstring))

    for num in newlist:
        dir_path = os.path.join(parent_directory, str(num))
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory {dir_path} created.")

    for imagefilename in os.listdir(image_directory):
        newstring = ""
        for char in imagefilename[:2]:
            if char.isnumeric():
                newstring += char
        targetdirectory = os.path.join(parent_directory, newstring)
        newimagepath = os.path.join(image_directory, imagefilename)
        shutil.move(newimagepath, targetdirectory)
        print(f"Image {imagefilename} moved to {targetdirectory}.")



def plot_number_of_images(file_path):
    dataset_directory = file_path
    class_counts = []

    for i in range(23):
        class_folder = os.path.join(dataset_directory, str(i))
        if os.path.isdir(class_folder):
            image_count = len([f for f in os.listdir(class_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
            class_counts.append(image_count)
        else:
            class_counts.append(0)

    class_labels = [str(i) for i in range(23)]

    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, class_counts, color='skyblue')
    plt.xlabel('Class Labels (0 to 22)')
    plt.ylabel('Number of Images')
    plt.title('Number of Images in Each Class')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

def even_out_class_size(dataset_directory, balanced_directory):
    os.makedirs(balanced_directory, exist_ok=True)

    class_counts = []
    class_folders = []

    for i in range(23):
        class_folder = os.path.join(dataset_directory, str(i))
        if os.path.isdir(class_folder):
            image_files = [f for f in os.listdir(class_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            class_counts.append(len(image_files))
            class_folders.append(class_folder)

    median_count = int(np.median(class_counts))
    target_count = median_count
    print(f"Median image count: {median_count}")

    final_class_counts = []
    for class_folder in class_folders:
        class_images = [f for f in os.listdir(class_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(class_images) > target_count:
            selected_images = random.sample(class_images, target_count)
            final_class_counts.append(target_count)
        else:
            selected_images = class_images
            final_class_counts.append(len(selected_images))

        target_class_folder = os.path.join(balanced_directory, os.path.basename(class_folder))
        os.makedirs(target_class_folder, exist_ok=True)
        for image in selected_images:
            shutil.copy(os.path.join(class_folder, image), os.path.join(target_class_folder, image))


def top_up_images(balanced_directory):
    end_image_count=400
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),                    
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.GaussianNoise(0.2)
    ])

    for class_folder in os.listdir(balanced_directory):
        target_class_folder = os.path.join(balanced_directory, class_folder)
        current_count = len(os.listdir(target_class_folder))
        
        while current_count < end_image_count:
            for image_name in os.listdir(target_class_folder):
                image_path = os.path.join(target_class_folder, image_name)
                image = tf.keras.utils.load_img(image_path)
                image = tf.keras.utils.img_to_array(image)
                
                augmented_image = data_augmentation(tf.expand_dims(image, axis=0))
                augmented_image = tf.squeeze(augmented_image).numpy()
                augmented_image_path = os.path.join(target_class_folder, f"aug_{current_count}_{image_name}")
                tf.keras.utils.save_img(augmented_image_path, augmented_image)
                current_count += 1
                
                if current_count >= end_image_count:
                    break


def load_and_remap_and_normalise_dataset(data_path, image_size=(128, 128), batch_size=256, seed=42, shuffle=True):
    lexicographical_order_list = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 3, 4, 5, 6, 7, 8, 9]

    data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        label_mode="int",
        shuffle=shuffle
    )

    def remap_labels(images, labels):
        new_labels = tf.gather(lexicographical_order_list, labels)
        return images, new_labels

    data = data.map(remap_labels)
    data=data.map(lambda x,y: (x/255,y))
    return data

def plot_loss_and_accuracy(hist):
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    axes[0].plot(hist.history['loss'], color='teal', label='Loss')
    axes[0].plot(hist.history['val_loss'], color='orange', label='Val Loss')
    axes[0].set_title('Loss', fontsize=16)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss Value')
    axes[0].legend(loc="upper left")
    
    # Plot Accuracy
    axes[1].plot(hist.history['accuracy'], color='teal', label='Accuracy')
    axes[1].plot(hist.history['val_accuracy'], color='orange', label='Val Accuracy')
    axes[1].set_title('Accuracy', fontsize=16)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy Value')
    axes[1].legend(loc="upper left")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def count_classes_in_sampled_batches(dataset, num_batches=9, plot=True):
    """
    Count the occurrences of each class in a sampled number of batches of the dataset.

    Parameters:
    - dataset: A TensorFlow dataset object containing images and labels.
    - num_batches: Number of batches to sample (default is 9).
    - plot: A boolean indicating whether to plot the counts (default is True).

    Returns:
    - batch_counts: A list of Counter objects with counts for each sampled batch.
    """
    batch_counts = []
    for i, (images, labels) in enumerate(dataset):
        if i >= num_batches:  
            break
        label_counts = Counter(labels.numpy())
        batch_counts.append(label_counts)

    for i, counts in enumerate(batch_counts):
        print(f"Batch {i + 1} counts: {dict(counts)}")

    if plot:
        all_classes = sorted(set().union(*[counts.keys() for counts in batch_counts]))
        
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
        axs = axs.flatten()

        for batch_index, counts in enumerate(batch_counts):
            # Prepare data for each subplot
            counts_values = [counts.get(cls, 0) for cls in all_classes]
            
            # Create a bar plot for the current batch
            axs[batch_index].bar(all_classes, counts_values, color='skyblue')
            axs[batch_index].set_title(f'Batch {batch_index + 1}')
            axs[batch_index].set_xlabel('Classes')
            axs[batch_index].set_ylabel('Count of Images')
            axs[batch_index].set_xticks(all_classes)  # Set x-ticks to class labels
            axs[batch_index].set_xticklabels(all_classes, rotation=45)

        # Adjust layout
        plt.tight_layout()
        plt.suptitle(f'Distribution of Classes Across {num_batches} Sampled Batches', fontsize=16)
        plt.subplots_adjust(top=0.9)  # Adjust top space to fit the suptitle

        plt.show()

    return batch_counts


def evaluate_model_for_specific_classes(model, test_dataset, class_1=6, class_2=10):
    """
    Evaluates the model on the test dataset and calculates false negative rates for the specified classes.

    Parameters:
        model: The trained model to evaluate.
        test_dataset: The dataset used for testing, yielding images and corresponding labels.
        class_1: The first class of interest.
        class_2: The second class of interest.

    Returns:
        dict: A dictionary with false negative rates for each specified class, test loss, and test accuracy.
    """
    y_true = []
    y_pred = []
    
    # Generate predictions and true labels
    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))  # Predicted labels
        y_true.extend(labels.numpy())  # True labels, assumed to be class indices
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    false_negative_rates = {}
    
    # Calculate false negatives for the specified classes
    for class_of_interest in [class_1, class_2]:
        false_negatives = np.sum((y_true == class_of_interest) & (y_pred != class_of_interest))
        total_positives = np.sum(y_true == class_of_interest)
        
        # Calculate false negative rate
        false_negative_rate = false_negatives / total_positives if total_positives > 0 else 0
        false_negative_rates[class_of_interest] = false_negative_rate
    
    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.4f}")

    return false_negative_rates, test_loss, test_acc


def get_misclassifications(model, dataset, class_names, verbose=1):
    misclassification_dicts={}
    all_predictions = []
    all_labels = []
    
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        all_predictions.extend(predictions)
        all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    pred_classes = np.argmax(all_predictions, axis=1)
    true_classes = all_labels.astype(int)
    
    misclassified_idx = np.where(pred_classes != true_classes)[0]
    
    misclassifications = []
    for idx in misclassified_idx:
        true_class_name = class_names[true_classes[idx]]
        pred_class_name = class_names[pred_classes[idx]]
        
        if true_class_name not in misclassification_dicts:
            misclassification_dicts[true_class_name] = {}
        if pred_class_name not in misclassification_dicts[true_class_name]:
            misclassification_dicts[true_class_name][pred_class_name] = 0
        misclassification_dicts[true_class_name][pred_class_name] += 1


        misclassifications.append([
            class_names[true_classes[idx]],  # true class
            class_names[pred_classes[idx]],  # predicted class
            f"{np.max(all_predictions[idx]) * 100:.2f}%"  # confidence
        ])
    
    if verbose == 1:
        print("Misclassification counts:")
        print(misclassification_dicts)
        print("\nDetailed misclassifications:")
        for item in misclassifications:
            print(item)
    
    return misclassifications