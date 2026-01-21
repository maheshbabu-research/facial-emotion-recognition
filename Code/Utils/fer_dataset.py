# FER Dataset pre-processing

import numpy as np                # Data wrangling.
import os                         # Manipulate operating system interfaces.
import pandas as pd               # Data handling.
import pickle                     # Python object serialization.
import plotly.express as px       # Data visualization
import plotly.graph_objects as go # Data visualization
import seaborn as sns             # Data visualization.
sns.set()
from sklearn.utils import resample

import matplotlib.pyplot as plt   # Data visualization.

import warnings                   # Ignore all warnings.
warnings.filterwarnings("ignore")


class FERDataset:
    def __init__(self):
        """
            Initializes the AI-ML Utils Class
        """
        super().__init__()

    def print_ds_statistics(self, root_folder, ds_file):
        """
            Prints the statistics of the dataset
            this is specifically for the FER dataset showing emotion category wise count
            and Usage (Training, PrivateTest, PublicTest) ratio
            and data imbalance

            Args:
            root_folder - path upto where the dataset file is present
            ds_file - name of the dataset file

            Returns:
            None
        """
        # construct the dataset file path
        ds_file_path = root_folder + ds_file
        
        # read the dataset file into a data frame
        df = pd.read_csv(ds_file_path)

        # print the shape of the data frame
        print("df.shape =", df.shape, "\n")

        # print the unique emotions in this data frame
        print("Unique emotions = ", sorted(df["emotion"].unique()), "\n")
        
        # print the total count of unique emotions
        print("# of Unique emotions =", len(df["emotion"].unique()), "\n")
        print(df.emotion.value_counts(), "\n")

        # print the unique usage (Training, PrivateTest, PublicTest)
        print("Unique Usage =", sorted(df["Usage"].unique()), "\n")
        
        # print the total unique usage
        print("# of Unique Usage =", len(df["Usage"].unique()), "\n")
        print(df.Usage.value_counts(), "\n")

        # print the mapping of the emotion index and its corresponding emotion
        print("0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral\n")

        # Count the number of images in each emotion category
        emotion_counts = df['emotion'].value_counts()

        # Print the counts
        print("Emotion Counts:")
        print(emotion_counts)

        # Calculate and print data imbalance statistics
        total_images = len(df)
        imbalance_stats = (emotion_counts / total_images) * 100

        print("\nData Imbalance Statistics (Percentage):")
        print(imbalance_stats.round(2))

        # view the dataset
        df

    # Function to oversample other than max class, given an upper limit
    def oversample_class(self, df, emotion, upperlimit):
        """
        Oversamples a specific class in the dataframe up to a specified upper limit.
    
       Args:
        df : pandas.DataFrame
            The input dataframe containing the data to be oversampled.
        
        emotion : str
            The class label (emotion) to be oversampled. 
        
        upperlimit : int
            The maximum number of samples that the specified class should have after oversampling
            (considering only the training set), additionally the PrivateTest and PublicTest
            might be present
    
        Returns:
        oversampled_class 
        """
    
        # extract the dataframe for the given emotion
        df_class = df[df['emotion'] == emotion]
        oversampled_class = resample(df_class, 
                                    replace=True,     # Sample with replacement
                                    n_samples= upperlimit - len(df_class),  # Number of samples to generate
                                    random_state=42)  # Reproducibility
        return oversampled_class

    def preprocess(self, root_folder, ds_source_file, ds_processed_file, max_class_samples_to_remove, min_classes_upper_limit):
        """
        Preprocesses the dataset by applying various transformations such as balancing class distributions and
        saving the processed dataset to a file.

        Args:
        root_folder : str
        The root directory where the dataset files are stored.

        ds_source_file : str
        The filename of the source dataset that needs to be processed. This file should be located in the `root_folder`.

        ds_processed_file : str
        The filename where the processed dataset will be saved after preprocessing. This file will be saved in the `root_folder`.

        max_class_samples_to_remove : int
        The maximum number of samples to remove from over represented classes in order to balance the dataset.

        min_classes_upper_limit : int
        The upper limit on the number of samples for classes that are under represented in the dataset. 
        The classes will be oversampled to reach this limit if needed.

        Returns:
        final_df: dataframe
        The final resampled dataframe to use
        resampled_file_path: str
        Path of the processed dataset file. The processed dataset is saved to `ds_processed_file`.
        
        """
        # Load the dataset
        ds_file_path = root_folder + ds_source_file
        df = pd.read_csv(ds_file_path)

        # Remove the Contempt emotion.
        df = df.loc[df["emotion"] != 7]

        # Separate the Training set
        train_df = df[df['Usage'] == 'Training']

        # Identify the majority class
        class_counts = train_df['emotion'].value_counts()
        max_class = class_counts.idxmax()

        print(max_class)

        # Number of samples to remove
        n = max_class_samples_to_remove  # Change this value to the desired number of samples to remove

        # Ensure n is not greater than the number of samples in the max class
        print( class_counts[max_class])
        if n > class_counts[max_class]:
            raise ValueError(f"Cannot remove {n} samples as it exceeds the number of samples in the majority class.")

        # Randomly remove n samples from the majority class
        df_max_class = train_df[train_df['emotion'] == max_class]
        df_max_class_removed = df_max_class.sample(n=len(df_max_class) - n, random_state=42)

        # Combine the reduced majority class with the rest of the training set
        train_df_reduced = pd.concat([train_df[train_df['emotion'] != max_class], df_max_class_removed])

        # Combine the reduced training set with the test sets
        reduced_df = pd.concat([train_df_reduced, df[df['Usage'] != 'Training']])



        # Identify the minority classes
        class_counts = reduced_df['emotion'].value_counts()
        min_classes = class_counts[class_counts < class_counts.max()].index


        # Oversample the minority classes
        oversampled_dfs = [self.oversample_class(reduced_df, emotion, upperlimit=min_classes_upper_limit) for emotion in min_classes]


        # Combine the oversampled classes with the original training set
        oversampled_train_df = pd.concat([reduced_df] + oversampled_dfs )

        # Combine the oversampled training set with the test sets
        final_df = pd.concat([oversampled_train_df, df[df['Usage'] != 'Training']])

        # Shuffle the dataset to ensure random distribution
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)


        # Save the new oversampled dataset to a CSV file
        resampled_file = ds_processed_file
        resampled_file_path = root_folder + resampled_file
        final_df.to_csv(resampled_file_path, index=False)

        print(f"Oversampled dataset saved to {resampled_file_path}")

        return final_df, resampled_file_path


     # Function to visualize an image from pixel data
    def visualize_image(self, pixels, emotion, usage):
        """
        Visualizes an image based on pixel data, displaying it along with emotion and usage labels.
    
        Args:
        pixels : str
            A string containing the pixel values of the image, in a space-separated format.
            The pixel values should represent a 48x48 grayscale image.
    
        emotion : str
            The label indicating the emotion associated with the image. 
            used in emotion recognition datasets.
    
        usage : str
            The label indicating how the image is used, such as 'Training', 'PublicTest', or 'PrivateTest'.
    
        Returns:
         None
            This function does not return a value. The image is displayed in a matplotlib window.
    
        """
        # Convert the space-separated pixel string into a NumPy array and reshape it into a 48x48 image
        image = np.array(pixels.split(), dtype='float32').reshape(48, 48)
        
        # Set the figure size for the plot
        plt.rcParams['figure.figsize'] = [2, 2]
        
        # Display the image using a grayscale color map
        plt.imshow(image, cmap='gray')
        
        # Set the title of the plot with emotion and usage information
        plt.title(f'Emotion: {emotion}, Usage: {usage}')
        
        # Remove the axis lines and labels
        plt.axis('off')
        
        # Show the image plot
        plt.show()


    def visualize_image(self, pixels, emotion, usage):
        """
        Visualizes a grayscale image based on pixel data and displays it with associated emotion and usage labels.
    
       Args:
        pixels : str
            A space-separated string containing the pixel values of the image.
            The pixels should represent a 48x48 grayscale image.
    
        emotion : str
            The label representing the emotion associated with the image (e.g., 'Happy', 'Sad'). 
    
        usage : str
            The label indicating how the image is categorized in the dataset (e.g., 'Training', 'Validation', or 'Test').
    
        Returns:
        None
           It displays the image in a matplotlib window.
        """
       # Convert the pixel data from a space-separated string to a numpy array of type float32
        # Then reshape the array to a 48x48 image
        image = np.array(pixels.split(), dtype='float32').reshape(48, 48)
        
        # Set the size of the figure for displaying the image (2x2 inches)
        plt.rcParams['figure.figsize'] = [2, 2]
        
        # Display the image using matplotlib with a grayscale colormap
        plt.imshow(image, cmap='gray')
        
        # Set the title of the plot to show the emotion and usage information
        plt.title(f'Emotion: {emotion}, Usage: {usage}')
        
        # Hide the axis labels and ticks for a cleaner image display
        plt.axis('off')
        
        # Show the image plot
        plt.show()


    def visualize_image_data(self, data_frame, count):
        """
        Visualizes a specified number of images from a given DataFrame, displaying them with their associated 
        emotion and usage labels.
    
        Args:
        data_frame : pandas.DataFrame
            The DataFrame containing the image data. It should have columns for pixel data, emotion labels, and usage labels.
        
        count : int
            The number of images to visualize from the DataFrame. The function will display the first 'count' images.
    
        Returns:
        None
            It displays the specified number of images using matplotlib
        """
        print("0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral\n")

        # Enumerate through each row and visualize the image
        counter = 0
        for index, row in data_frame.iterrows():
            self.visualize_image(row['pixels'], row['emotion'], row['Usage'])
            counter = counter + 1
            
            # Break after displaying images upto specified count
            if counter > count:
                break



