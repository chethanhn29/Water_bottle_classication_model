# Water Bottle Image Classication Using CNN
![download (7)](https://user-images.githubusercontent.com/110838853/211126090-069a3981-b107-4e91-81a7-caafc5485650.jpg)  ![images (9)](https://user-images.githubusercontent.com/110838853/211126006-8baad9cf-85fb-4f8a-9ba2-c3b7e3022742.jpg)                 ![download (8)](https://user-images.githubusercontent.com/110838853/211126092-e2025ae4-52e4-453a-94c6-2ea6ba52e966.jpg)                   ![download (9)](https://user-images.githubusercontent.com/110838853/211126169-2cb27b26-f30c-4ba6-9692-0032b14dbdb8.jpg)    






## Objective: 
image classification model to classify water bottles as Full,Half,Overflowing Water bottles based on water level inside

## Data Collection,Preprocessing And Model training and Deployment

#### 1.Collect a labeled dataset of images of water bottles, with each image labeled with the corresponding water level (full, half, or overflowing).
This dataset will be used to train and evaluate the CNN model.

Consider collecting a diverse and representative set of images to improve the model's generalizability.
#### 2.Preprocess the images as needed, such as by resizing or cropping them.
Preprocessing the images can help to ensure that they are in a consistent format and ready for use as input to the CNN.

Common preprocessing techniques include resizing, cropping, and normalization.

#### 3.Use data augmentation techniques to increase the size of the dataset and improve model generalization.

Data augmentation involves generating additional training data by applying transformations to the existing data.

This can help to prevent overfitting and improve the model's performance on new, unseen data.
#### 4.Convert the images to a size of 256x256 and ensure that they have 3 channels (RGB).
The CNN model will expect input images to be in a specific format, such as a certain size and number of channels.

Converting the images to the required format is an important step in preparing the data for use as input to the model.
#### 5.Split the dataset into training and validation sets.
The training set will be used to train the CNN model, while the validation set will be used to evaluate the model's performance during training.

A common split is to use 80% of the data for training and 20% for validation.
#### 6.Implement a CNN using TensorFlow and Keras.
TensorFlow is a popular machine learning library that can be used to implement and train neural networks.

Keras is a high-level API for building and training neural networks that runs on top
#### 7.Use the Adam optimizer and a batch size of 32 to train the model on the training set.
The Adam optimizer is a popular choice for training neural networks, as it can adapt the learning rate for each parameter during training.

The batch size determines the number of samples to work through before the model's weights are updated. 

A larger batch size can lead to faster training times, but may also result in less stable gradients.
#### 8.Use class weights to address imbalanced classes in the dataset.
If the dataset is imbalanced, with some classes being much more common than others, the model may be biased towards the more common classes.

Class weights can be used to balance the loss function, so that the model pays more attention to the less common classes.

Monitor the model's performance on the validation set during training.
#### 9.Use the validation set to evaluate the model's performance as it is being trained.

This can help to identify overfitting and ensure that the model is generalizing well to new, unseen data.

#### 10.Calculate performance metrics such as precision, recall, validation loss, and accuracy loss.
Performance metrics can provide insights into the model's strengths and weaknesses and help to identify areas for improvement.

Precision and recall are useful for evaluating the model's ability to classify samples correctly.

Validation loss and accuracy loss can be used to monitor the model's performance on the validation set and identify overfitting.

#### 11.Deploy the model using Gradio for real-time classification.
Gradio is a library that makes it easy to deploy machine learning models for real-time use.

Use Gradio to create a user interface for the model, allowing users to input images and receive predictions from the model.
#### 12.Use the trained model to classify new images of water bottles and predict the water level inside.
Once the model has been trained and deployed, it can be used to classify new images of water bottles and predict the water level inside.

This can be useful for automating the process of monitoring water levels, reducing the need for manual inspection.

The model can be used with the user interface created in step 11 to allow users to input images and receive predictions in real-time.
