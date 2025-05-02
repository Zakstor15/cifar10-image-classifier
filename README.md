# CIFAR-10 Image Classification using ANN & CNN

This project demonstrates how to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) using two deep learning models:
- **Artificial Neural Network (ANN)**
- **Convolutional Neural Network (CNN)**

The models are built and trained using TensorFlow/Keras and evaluated on test data.

## 🧠 Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes:
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

This project includes:
- Data preprocessing and normalization
- ANN model with fully connected dense layers
- CNN model with convolutional and pooling layers
- Model evaluation using classification report and accuracy
- Visualization of predictions on test samples

## 📦 Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies with:
```bash
pip install tensorflow matplotlib seaborn scikit-learn

🚀 Getting Started

Clone this repo and run the notebook or script:

git clone https://github.com/Mueez-lab/cifar10-image-classifier.git
cd cifar10-image-classifier

Run the Python notebook:

jupyter notebook

Or run the Python file:

python main.py

📊 Model Comparison
Model	Accuracy
ANN	~55-60%
CNN	~70-75%

    Note: CNN performs better as it captures spatial features using convolutional layers.

📸 Sample Prediction

You can visualize predictions like this:

plot_sample_with_prediction(x_test, index=1, model=cnn)

This will show the image and its predicted class.
📁 Project Structure

.
├── cifar10_ann_cnn.py  # or .ipynb if using a notebook
├── README.md
└── requirements.txt    # (optional)

📚 References

    CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

    TensorFlow Documentation: https://www.tensorflow.org/

📝 License

This project is open source under the MIT License.
