# AI-ML Utils for standardized reporting of model performance and metrics
import itertools
from os import getcwd
from sys import path


import matplotlib.pyplot as plt   # Data visualization.
import numpy as np                # Data wrangling.
import os                         # Manipulate operating system interfaces.
import pandas as pd               # Data handling.
import pickle                     # Python object serialization.
import plotly.express as px       # Data visualization
import plotly.graph_objects as go # Data visualization
import seaborn as sns             # Data visualization.
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score


# pyplot for visualization accuracy 
from matplotlib import pyplot as plt

# For confusion matrix, f1-score etc
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from sklearn.model_selection import train_test_split

import warnings                   # Ignore all warnings.
warnings.filterwarnings("ignore")


class AIMLUtils:
    '''
    AI-ML utility functions for standardized visualize, score
    '''
    def __init__(self):
        """
            Initializes the AI-ML Utils Class
        """
        super().__init__()

    def calc_f1_score(self, y_test, y_pred):
        """
        Calculates the F1-Score given the y_test and y_pred with weighted average

        Args:
            y_true (list or numpy.ndarray): The true labels.
            y_pred (list or numpy.ndarray): The predicted labels.

        Returns:
            f1 score
         """
        # Calculate F1-Score
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        print(f'F1-Score: {f1:.2f}')
        return f1

    def plot_confusion_matrix(self, y_true, y_pred, title):
        """
        Plots the confusion matrix for the given true and predicted labels.

        Args:
            y_true (list or numpy.ndarray): The true labels.
            y_pred (list or numpy.ndarray): The predicted labels.
            title (str): The title of the plot.

        Returns:
            None
         """
        # Convert the predicted probabilities to class labels by taking the index of the maximum value along axis 1
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Compute the confusion matrix comparing the true labels (y_true) with the predicted class labels (y_pred_classes)
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Create a new figure for the plot with a specified size (8x6 inches)
        plt.figure(figsize=(8, 6))
        
        # Display the confusion matrix using Matplotlib with a 'Blues' colormap for better visual distinction
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Set the title of the confusion matrix plot
        plt.title(title)
        
        # Add a colorbar to indicate the scale of the confusion matrix values
        plt.colorbar()
        
        # Label the x-axis as 'Predicted labels'
        plt.xlabel('Predicted labels')
        
        # Label the y-axis as 'True labels'
        plt.ylabel('True labels')
        
        # Iterate over each cell in the confusion matrix
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # Add text labels to each cell of the confusion matrix plot
            # The text color is white if the value is greater than half the maximum value in the confusion matrix;
            #otherwise, it's black
            plt.text(j, i, cm[i, j], horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        # Display the confusion matrix plot
        plt.show()



    
    def print_classification_report(self, y_test, y_pred):
        """
        Print the classification report for the target emotions

        Args:
            y_true (list or numpy.ndarray): The true labels.
            y_pred (list or numpy.ndarray): The predicted labels.
        Returns:
            None
         """
        # Classification report
        from sklearn.metrics import classification_report
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        y_pred_classes = np.argmax(y_pred, axis=1)
        print(classification_report(y_test, y_pred_classes, target_names=emotion_labels))


    def plot_training_history(self, history, model_title) :
        """
        Plots the training and validation loss and accuracy metrics over epochs for a given model.
        Creates two plots:
           1. Training and validation loss
           2. Training and validation accuracy

        Helps in analyzing model's performance and issues such as overfitting or underfitting

        Args:
            history (keras.callbacks.History):
            A Keras History object containing the training metrics. 
            model_title (str): A string representing the title of the plots. 

        Returns:
            None
        """
        # Plot training accuracy over epochs
        plt.plot(history.history['accuracy'], label='Train Accuracy')  
        
        # Plot the validation accuracy
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  
        
        # Label the x-axis as 'Epochs'
        plt.xlabel('Epochs')
        
        # Label the y-axis as 'Accuracy'
        plt.ylabel('Accuracy')
        
        # Set the title of the plot to show the comparison of training and validation accuracy
        plt.title(model_title + ' Training vs Validation Accuracy')
        
        # Add a legend to the plot to differentiate between training and validation accuracy
        plt.legend()

        plt.figure(figsize=(9, 6)) 
        
        # Display the plot
        plt.show()
        
        # Plot training loss over epochs
        plt.plot(history.history['loss'], label='Train Loss')  
        
        # Plot the validation loss values
        plt.plot(history.history['val_loss'], label='Validation Loss')  
        
        # Label the x-axis as 'Epochs'
        plt.xlabel('Epochs')
        
        # Label the y-axis as 'Loss'
        plt.ylabel('Loss')
        
        # Set the title of the plot to show the comparison of training and validation loss
        plt.title(model_title + ' Training vs Validation Loss')
        
        # Add a legend to the plot to differentiate between training and validation loss
        plt.legend()

        plt.figure(figsize=(9, 6)) 
        
        # Display the plot
        plt.show()


    def print_model_summary(self, model):
        """
        Prints a detailed summary of the Keras model architecture.
        Summary includes info on each layer in the model like
        type, output shape, number of parameters, to number of parameters in the model.

        Useful for understanding the structure of the model and verifying its configuration
 
        Args:
            model (keras.Model): 
            A Keras model instance for which the summary is to be printed. 
            
        Returns:
            None
        """
        # Print the summary of the Keras model, which includes details about each layer and overall parameters
        model.summary()

