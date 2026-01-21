import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt


class FERCustomCNN:

    # Build a lightweight CNN model
    def create_model(self):
        '''
        Create a lightweight CNN model for FER (Facial Emotion Recognition
        Args: None
        Returns:
        CNN model to be used for training / validation
        '''
        # Initialize a sequential model
        # Build CNN layers to recognition features in the image starting from course to finer details
        model = keras.Sequential([
            # Convolutional layer with 32 filters, each of size 3x3, using ReLU activation function
            # Input shape is (48, 48, 1), meaning images are 48x48 pixels with 1 channel (grayscale)
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            
            # Max pooling layer with pool size of 2x2, reducing the spatial dimensions by half
            keras.layers.MaxPooling2D((2, 2)),
            
            # Second convolutional layer with 64 filters, each of size 3x3
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Second max pooling layer with pool size of 2x2
            keras.layers.MaxPooling2D((2, 2)),
            
            # Third convolutional layer with 128 filters, each of size 3x3
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            
            # Third max pooling layer with pool size of 2x2
            keras.layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional layer with 256 filters, each of size 3x3
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            
            # Fourth max pooling layer with pool size of 2x2
            keras.layers.MaxPooling2D((2, 2)),
            
            # Flatten layer to convert the 3D output of the convolutional layers into a 1D vector
            keras.layers.Flatten(),
            
            # Fully connected (dense) layer with 256 neurons and ReLU activation function
            keras.layers.Dense(256, activation='relu'),
            
            # Output layer with 7 neurons (one for each emotion) and softmax activation function to produce probability distributions
            keras.layers.Dense(7, activation='softmax')  # Output layer for 7 emotions
        ])
    
        # Return the compiled model
        return model

    # create and compile model
    def create_and_compile_model(self):
        '''
        Create an compile the model
        Args: None
        Returns:
        Compiled Model
        '''
        model = self.create_model()

        # Compile the model with the following configuration:
        # - Optimizer: 'adam' is used for efficient gradient-based optimization.
        # - Loss function: 'sparse_categorical_crossentropy' is chosen for multi-class classification tasks
        #   where the target labels are integers.
        # - Metrics: 'accuracy' is used to monitor the performance of the model during training and evaluation.

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Preprocess images

    def load_images(self, data_frame):
        """
        Converts pixel data from a DataFrame into a NumPy array of images.
    
        Args:
        data_frame : pandas.DataFrame
            The DataFrame containing the image data. It should have a column named 'pixels', 
            each entry is a space-separated string representing the pixel values of a 48x48 grayscale image.
    
        Returns:
        --------
        np.ndarray
            A NumPy array of images, where each image is represented as a 48x48 array of pixel values.
        """
    
        images = []
    
        # Iterate over each row in the DataFrame
        for index, row in data_frame.iterrows():
            pix = row['pixels']
            
            # Convert the space-separated pixel string into a NumPy array and reshape it into a 48x48 image
            image = np.array(pix.split(), dtype='float32').reshape(48, 48)
            
            # Append the image to the list of images
            images.append(image)
        
        # Convert the list of images to a NumPy array before returning
        return np.array(images)


    def train_model(self, data_frame, model, test_size, epochs=20, batch_size=25, validation_split=0.20):
        """
        Trains a Convolutional Neural Network (CNN) model on a dataset of images.
    
       Args:
        data_frame : pandas.DataFrame
            The DataFrame containing the image data and associated labels. It should have a 'pixels' column for the image data 
            and an 'emotion' column for the labels.
    
        model : keras.Model
            The CNN model to be trained.
    
        test_size : float
            The proportion of the dataset to include in the test split (e.g., 0.2 for 20%).
    
        epochs : int, optional
            The number of epochs to train the model for. Default is 20.
    
        batch_size : int, optional
            The number of samples per gradient update. Default is 25.
    
        validation_split : float, optional
            The fraction of the training data to be used as validation data. Default is 0.20.
    
        Returns:
        tuple:
            - X_test (np.ndarray): The test set images, ready for evaluation.
            - y_test (np.ndarray): The corresponding labels for the test set.
            - history (keras.callbacks.History): The training history, including training and validation loss and accuracy.
            - label_encoder (LabelEncoder): The encoder used for transforming the emotion labels into integers.
        """
    
        # Load and preprocess the images
        X = self.load_images(data_frame)
        X = X.astype('float32') / 255.0  # Normalize pixel values to the range [0, 1]
        
        # Extract emotion labels
        y = data_frame['emotion'].values
    
        # Encode emotion labels into integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
    
        # Reshape the data for the CNN model by adding a channel dimension
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
    
        # Train the model
        history = model.fit(X_train, y_train, epochs=20, batch_size=25, validation_split=0.20)
        return X_test,y_test, history, label_encoder

    def evaluate_model(self, model, X_test):
        """
        Evaluates a trained Convolutional Neural Network (CNN) model on a test dataset.
    
        Args:
        model : keras.Model
            The trained CNN model to be evaluated.
    
        X_test : np.ndarray
            The test set images, preprocessed and ready for evaluation. The data should be in the same format 
            as used during training (e.g., with a channel dimension).
    
        Returns:
        np.ndarray
            The predicted classes for the test set. Each entry corresponds to the predicted label for a test image.
        """
    
        # Predict the probabilities for each class using the test set
        y_pred = model.predict(X_test)
        
        # Convert the predicted probabilities to class labels by taking the argmax across the class axis
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Return the predicted class labels
        return y_pred
