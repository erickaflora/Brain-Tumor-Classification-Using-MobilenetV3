# Brain Tumor Classification Using MobilenetV3
This project focuses on classifying brain tumor MRI images into categories such as glioma, meningioma, and pituitary tumor. MobileNetV3, a lightweight and efficient convolutional neural network, is used to achieve high accuracy while maintaining computational efficiency.

## Requirements
+ Python >= 3.8 <= 3.11
+ pytorch
+ matplotlib
+ scikit-learn
+ tqdm

+ You can install the required packages using:
```bash
pip install torch torchvision matplotlib scikit-learn tqdm
```
## Dataset
The dataset used in this project is the [Brain tumor multimodal image (CT & MRI) Dataset](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri). The MRI dataset subfolders were initially unbalanced, so I removed some images to create a balanced dataset. After balancing, the dataset was split into `train` and `validation` folders, with each folder containing subfolders for each class.
The data were preprocessed by applying transformations and data augmentation.

## Training
The model was trained for 50 epochs with the following hyperparameters:
+ Learning rate: 0.0001
+  Batch size: 32
+  num_workers: 8
+  Optimizer: Adam
+  Loss function: CrossEntropyLoss
Training was performed on an NVIDIA RTX 3060 Ti GPU.

## Evaluation
The model achieves the following performance on the validation set:
+ Valid Loss: 0.0204
+ Valid Accuracy: 93.33%
+ F1 Scores:
  + Glioma: 0.91
  + Meningioma: 0.91
  + Pituitary: 0.98
+ The model generalizes well across all classes given that it achieved an accuracy of 93% and similar F1 scores.
 
## Acknowledgments
+ [Brain tumor multimodal image (CT & MRI) Dataset](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri) for providing the data.
+ The PyTorch team for their deep learning framework.
