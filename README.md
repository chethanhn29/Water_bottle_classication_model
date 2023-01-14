# Water Bottle Image Classication Using CNN
  ![glass-bottle-different-water-levels-14101289](https://user-images.githubusercontent.com/110838853/211681792-09189c4d-567b-4a9d-8857-c0979eecbfbc.jpg)

## Objective: 
image classification model to classify water bottles as Full,Half,Overflowing Water bottles based on water level inside

## Data Collection,Preprocessing And Model training and Deployment

#### 1.   Collect a labeled dataset of images of water bottles, with each image labeled with the corresponding water level (full, half, or overflowing).
This dataset will be used to train and evaluate the CNN model.

Consider collecting a diverse and representative set of images to improve the model's generalizability.

The images in this dataset were obtained [Water Bottle Dataset](https://www.kaggle.com/datasets/chethuhn/water-bottle-dataset)
from Kaggle.
#### 2.   Preprocess the images as needed, such as by resizing or cropping them.
Preprocessing the images can help to ensure that they are in a consistent format and ready for use as input to the CNN.

Common preprocessing techniques include resizing, cropping, and normalization.

![1_wYu4oK_TgFUvdlH5kq-Axw](https://user-images.githubusercontent.com/110838853/212464192-26a72a0a-1777-435c-a560-df02e641a83c.png)


#### 3. Use data augmentation techniques to increase the size of the dataset and improve model generalization.

Data augmentation involves generating additional training data by applying transformations to the existing data.

This can help to prevent overfitting and improve the model's performance on new, unseen data.
#### 4. Convert the images to a size of 256x256 and ensure that they have 3 channels (RGB).
The CNN model will expect input images to be in a specific format, such as a certain size and number of channels.

Converting the images to the required format is an important step in preparing the data for use as input to the model.
#### 5. Split the dataset into training and validation sets.
The training set will be used to train the CNN model, while the validation set will be used to evaluate the model's performance during training.




### Folder Structure
#### data: This folder contains the dataset of water bottle images that is used to train and evaluate the model. The images are labeled with the corresponding water level (full, half, or overflowing).
#### model: This folder contains the trained model and the code for training the model.
notebook: This folder contains the Jupyter notebook used for preprocessing the data, building and training the model.
Documentation
#### Guide.pdf: This documentation provides a detailed guide on how to use the model and how to train and evaluate the model on new data.
You can also provide the link of the dataset or the model that you used in the project and also provide the link of the documentation if it is available online.

For example:

The dataset used in this project can be found at Water Bottle Dataset
The trained model can be found at Water Bottle Model
The documentation can be found at Guide

A common split is to use 80% of the data for training and 20% for validation.

#### 6. Implement a CNN using TensorFlow and Keras.
TensorFlow is a popular machine learning library that can be used to implement and train neural networks.
Keras is a high-level API for building and training neural networks that runs on top



#### 7. How the Image data works in my Model 
![jmV3oAz](https://user-images.githubusercontent.com/110838853/212463553-57589f0e-b27c-448a-8bb7-9d2c4f6eb729.jpg)
a)Input: The model takes in images of water bottles as input. These images are typically preprocessed and resized to a fixed size (e.g. 256x256 pixels) before being fed into the model.

b)Convolutional Layers: The model then applies a series of convolutional layers to extract features from the input images. These layers use filters (also called kernels or weights) to detect patterns in the images, such as edges or textures. The filters are typically small (e.g. 3x3 pixels) and move across the entire image, creating a feature map for each filter. The number of filters in each layer and their stride (i.e. how much they move) can be adjusted to control the complexity of the model.

c)Pooling Layers: The model also applies pooling layers to the feature maps, which down-sample the maps by taking the maximum or average of a group of adjacent pixels. This reduces the dimensionality of the data and helps to make the model more robust to small variations in the input images.

d)Fully Connected Layers: After the convolutional and pooling layers, the model flattens the feature maps and applies a series of fully connected layers (also called dense layers) to make the final predictions. These layers use weights to combine the features from the previous layers and produce a probability for each class. The number of neurons in these layers and their activation functions can be adjusted to control the model's capacity.

e)Output: The model produces a probability for each class, which can be converted into a predicted class using a classifier such as argmax or softmax. The model can also be fine-tuned using the loss function and optimizer.

Overall, the model takes in an image of a water bottle and applies a series of convolutional and pooling layers to extract features from the image. These features are then passed through fully connected layers to make the final prediction. This process is known as feature extraction and it is the backbone of CNNs.




#### 8. Use the Adam optimizer and a batch size of 32 to train the model on the training set.
The Adam optimizer is a popular choice for training neural networks, as it can adapt the learning rate for each parameter during training.

The batch size determines the number of samples to work through before the model's weights are updated. 

A larger batch size can lead to faster training times, but may also result in less stable gradients.

#### 9. Use class weights to address imbalanced classes in the dataset.
If the dataset is imbalanced, with some classes being much more common than others, the model may be biased towards the more common classes.

Class weights can be used to balance the loss function, so that the model pays more attention to the less common classes.

Monitor the model's performance on the validation set during training.

#### 10. Use the validation set to evaluate the model's performance as it is being trained.

This can help to identify overfitting and ensure that the model is generalizing well to new, unseen data.

#### 11.  Calculate performance metrics such as precision, recall, validation loss, and accuracy loss.
Performance metrics can provide insights into the model's strengths and weaknesses and help to identify areas for improvement.

Precision and recall are useful for evaluating the model's ability to classify samples correctly.

Validation loss and accuracy loss can be used to monitor the model's performance on the validation set and identify overfitting.

#### 12.  Deploy the model using Gradio for real-time classification.
Gradio is a library that makes it easy to deploy machine learning models for real-time use.

Use Gradio to create a user interface for the model, allowing users to input images and receive predictions from the model.
#### 13.  Use the trained model to classify new images of water bottles and predict the water level inside.
Once the model has been trained and deployed, it can be used to classify new images of water bottles and predict the water level inside.

This can be useful for automating the process of monitoring water levels, reducing the need for manual inspection.

The model can be used with the user interface created in step 11 to allow users to input images and receive predictions in real-time.


## Folder Structre 

### [Dataset](https://github.com/chethanhn29/Water_bottle_classication_model/tree/main/Dataset)  
This folder contains the labeled dataset of images of water bottles that was used to train and evaluate the CNN model. The images are labeled with the corresponding water level (full, half, or overflowing).

### [Model](https://github.com/chethanhn29/Water_bottle_classication_model/tree/main/Model)
This folder contains the trained CNN model and any associated files, such as the model architecture, weights, and training history.

### [Deployed Model Predictions](https://github.com/chethanhn29/Water_bottle_classication_model/tree/main/Deployed%20Model%20Predictions)
This folder contains example predictions made by the deployed model on new, unseen data.

### [Source Code](https://github.com/chethanhn29/Water_bottle_classication_model/tree/main/Source%20Code)
This folder contains the source code for this project, including the code for data preprocessing, model training and evaluation, and model deployment.

### [Documentation](https://github.com/chethanhn29/Water_bottle_classication_model/tree/main/Documentation)
This folder contains any additional documentation for this project, such as project reports, presentations, and user guides.
