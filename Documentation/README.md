
# Water Bottle Image Classification Using CNN

## Objective
The objective of this project is to build and train a Convolutional Neural Network (CNN) model for classifying images of water bottles based on the water level inside (full, half, or overflowing). The model is implemented using TensorFlow and Keras, and is trained on a labeled dataset of images. Data augmentation techniques are used to increase the size of the dataset and improve model generalization. The model is evaluated using performance metrics such as precision, recall, validation loss, and accuracy. The model is also deployed using Gradio for real-time classification.

## Requirements
1. TensorFlow
2. Keras
3. Gradio
4. Numpy
5. Matplotlib


## Data Collection and Preprocessing:

1. The first step in this project is to collect a labeled dataset of images of water bottles, with each image labeled with the corresponding water level (full, half, or overflowing). You can use the water bottle dataset available at [Kaggle](https://www.kaggle.com/datasets/chethuhn/water-bottle-dataset), 
or you may choose to create your own dataset by manually labeling images or using a tool such as Labelbox.

2. Once the dataset has been downloaded or created, it is important to preprocess the images as needed to ensure that they are all of the same size and have the same number of channels (e.g. 3 for RGB images). This can be done using functions such as resize, crop, or pad.

3. To further increase the size of the dataset and improve model generalization, data augmentation techniques can be applied to the images. This can include techniques such as random horizontal and vertical flips, random rotations, and other transformations. These techniques can be applied using tools such as TensorFlow's image_dataset_from_directory method or Keras' ImageDataGenerator class.

4. After preprocessing and data augmentation, the images should be converted to a size that is suitable for your model, such as 256x256. It is also important to ensure that the images have the correct number of channels (e.g. 3 for RGB images).

5. Finally, the dataset should be split into training, validation, and test sets using a function such as train_test_split or sklearn's model_selection.train_test_split. A typical split might be 70% for training, 10% for validation, and 20% for testing, but this can vary depending on the size and quality of your dataset. It is also a good idea to shuffle the data before splitting it to avoid any biases or patterns in the order of the samples.

6. Caching and prefetching the datasets can improve performance during training by reducing the time spent loading and processing data. This can be done using TensorFlow's cache and prefetch functions.


## Model Implementation and Training:

   To implement the CNN model for this project, you will need to follow the steps below:

1.  Collect a labeled dataset of images of water bottles, with each image labeled with the corresponding water level (full, half, or overflowing). This dataset will be used to train and evaluate the CNN model.

2.  Preprocess the images as needed, such as by resizing or cropping them. Preprocessing the images can help to ensure that they are in a consistent format and ready for use as input to the CNN. Common preprocessing techniques include resizing, cropping, and normalization.

3.  Use data augmentation techniques to increase the size of the dataset and improve model generalization. Data augmentation involves generating additional training data by applying transformations to the existing data. This can help to prevent overfitting and improve the model's performance on new, unseen data.

4.  Convert the images to a size of 256x256 and ensure that they have 3 channels (RGB). The CNN model will expect input images to be in a specific format, such as a certain size and number of channels. Converting the images to the required format is an important step in preparing the data for use as input to the model.

5.  Split the dataset into training and validation sets. The training set will be used to train the CNN model, while the validation set will be used to evaluate the model's performance during training. A common split is to use 80% of the data for training and 20% for validation.

6.  Implement a CNN using TensorFlow and Keras. TensorFlow is a popular machine learning library that can be used to implement and train neural networks. Keras is a high-level API for building and training neural networks that runs on top of TensorFlow.

7.  Use the Adam optimizer and a batch size of 32 to train the model on the training set. The Adam optimizer is a popular choice for training neural networks, as it can adapt the learning rate for each parameter during training. The batch size determines the number of samples to work through before the model's weights are updated. A larger batch size can lead to faster training times, but may also result in less stable gradients.

8.  Use class weights to address imbalanced classes in the dataset. If the dataset is imbalanced, with some classes having significantly more examples than others, you can use class weights to give more weight to the underrepresented classes. This can help the model to better learn and classify the underrepresented classes.

9.  Monitor the model's performance on the validation set during training, using metrics such as accuracy and loss. If the model is not performing well on the validation set, you may need to adjust the model architecture, the optimizer parameters, or the data preprocessing and augmentation techniques.
    Once the model is performing well on the validation set, you can train it on the full dataset and save the trained model for future use.
    
    
 ## Model Architecture:

The model architecture for this project is a Convolutional Neural Network (CNN) designed for image classification tasks. The CNN consists of multiple layers that work together to learn and extract features from the input images, and use these features to classify the images into different categories.

The specific layers of the CNN include:

1.  Input layer: This layer receives the input images and passes them through the rest of the network.
2.  Convolutional layers: These layers apply a convolution operation to the input images, which involves sliding a filter over the image and computing a dot product between the filter and the image at each position. This helps the model to learn local patterns and features in the images. The model has 2 Conv2D layers with 32 and 64 filters, respectively, and a kernel size of 3x3.
3.  Pooling layers: These layers downsample the output of the convolutional layers, reducing the spatial dimensions of the feature maps and making the model more efficient. The model has 2 MaxPooling2D layers with pool sizes of 2x2.
4.  Flatten layer: This layer flattens the output of the pooling layers into a single long vector, which can then be used as input to the dense layers.
5.  Dense layers: These layers are fully connected and allow the model to learn more global patterns in the data. The model has 2 Dense layers with 128 and 64 units, respectively, and a final Dense layer with 3 output units and a softmax activation function.
6.  Output layer: This layer produces the final classification predictions for the input images.
7.  The model is trained using the Adam optimizer and a categorical cross-entropy loss function. The model's performance is evaluated using accuracy and loss metrics.

## Deployment and Classification:

1.  The trained model can be deployed for real-time classification using a tool such as Gradio. Gradio is a library that allows you to create interactive web interfaces for your machine learning models, allowing users to input data and receive predictions in real time.
2.  To deploy the model using Gradio, you will need to install the library and import it in your code. You will also need to define a function that takes in input data and returns predictions based on the trained model.
3.  Once the function is defined, you can use Gradio to create an interactive interface that allows users to input data (in this case, images of water bottles) and receive predictions from the model.
4.  To classify new images using the trained model, you can use the predict method of the model object. This method takes in an input tensor and returns a tensor of predictions for each sample in the input. You can then use these predictions to classify the images into the appropriate water level categories.


## Code Breakdown:

The code for this project is organized into several main sections:

1.  Import necessary libraries and modules, including tensorflow and matplotlib.
2.  Set the batch size, image size, number of channels, and number of epochs to use during training.
3.  Load the dataset from a directory using tf.keras.preprocessing.image_dataset_from_directory.
4.  Get the class names for the dataset.
5.  Take a sample batch of images and labels from the dataset and print the shape of the batch and the labels.
6.  Use matplotlib to display the images in the batch along with their labels.
7.  Calculate the total size of the dataset and determine the sizes of the training, validation, and test sets using a combination of len and take or skip methods.
8.  Define a function get_dataset_partitions_tf to split the dataset into train, validation, and test sets, with the option to shuffle the data and set the sizes of each set.
9.  Use the function to split the dataset and assign the resulting datasets to train_ds, val_ds, and test_ds.
10. Define a tf.keras.Sequential model for preprocessing the images, consisting of a Resizing layer and a Rescaling layer.
11. Define another neural.keras.Sequential model for data augmentation, consisting of a RandomFlip layer and a RandomRotation layer.
12. Apply data augmentation to the training set using the data_augmentation model.
13. Define a CNN model using the functional API of Keras. This model consists of 2 Conv2D layers, 2 MaxPooling2D layers, a Flatten layer, and 2 Dense layers, followed by an output layer with a softmax activation function.
14. Compile the model using the Adam optimizer and a Sparse Categorical Crossentropy loss function.
15. Use class weights to address imbalanced classes in the dataset.
16. Train the model on the training set using the fit method. Validate the model on the validation set using the validate method.
17. Save the trained model to a file using the save method.
18. Use Gradio to create an interactive interface for real-time classification using the model.
19. Define a function for classifying new images using the trained model.
20. Use the function to classify new images and print the resulting predictions.


 
    
    
 ## Future Work
There are several areas for future improvement in this project:

1.  Increase the size of the dataset by collecting and labeling more images.
2.  Explore the use of different model architectures and hyperparameters to improve performance.
3.  Implement additional data augmentation techniques.
4.  Use the test set to evaluate the model's performance on unseen data.
5.  Deploy the model to a production environment, such as a web server or mobile app.
 
## Conclusion
This project demonstrates the use of a CNN model for classifying images of water bottles based on the water level inside. The model is implemented using TensorFlow and Keras, and is trained on a labeled dataset of images. Data augmentation techniques are used to increase the size of the dataset and improve model generalization. The model is evaluated using performance metrics such as precision, recall, validation loss, and accuracy. The model is also deployed using Gradio for real-time classification. There are several opportunities for future work to further improve the model's performance and deployment.




Model Architecture
The CNN model implemented in this repository consists of the following layers:

2 Conv2D layers with 32 and 64 filters, respectively, and a kernel size of 3x3
2 MaxPooling2D layers with pool sizes of 2x2
A Flatten layer
2 Dense layers with 128 and 64 units, respectively
A final Dense layer with 3 output units and a softmax activation function
The model is compiled with the Adam optimizer and a Sparse Categorical Crossentropy loss function, and is trained using a batch size of 32. Class weights are used to address imbalanced classes in the dataset.

Evaluation and Deployment
To evaluate the performance of the model, several metrics are calculated on the validation set after training. These metrics include precision, recall, validation loss, and accuracy.

The model is also deployed using Gradio for real-time classification. This allows users to classify new images of water bottles and predict the water level inside by simply uploading an image or providing a URL to the image.
