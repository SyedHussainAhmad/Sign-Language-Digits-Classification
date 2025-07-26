# Sign Language Digit Classification with Transfer Learning

This project focuses on classifying hand gestures representing digits (0–9) using deep learning models implemented in **PyTorch**. It utilizes the **Sign Language Digits Dataset**, which contains RGB images of hand signs corresponding to digits 0 through 9.

---

## Models Implemented

### 🔹 Fully Connected Neural Network (FCNN)
- **Input**: Flattened RGB images (3×64×64 = 12,288 features)  
- **Architecture**: 3-layer feedforward neural network  
- **Activation**: ReLU  
- **Output**: 10-class softmax  

### 🔹 Convolutional Neural Network (CNN)
- **Layers**:  
  - Two convolutional layers with ReLU activation and max pooling  
  - Fully connected layers for classification  
- **Input**: 64×64 RGB images  
- **Output**: 10-class softmax  

### 🔹 ResNet-18 (Transfer Learning)
- **Pretrained on**: ImageNet  
- **Configurations**:  
  - **ResNet-Frozen**: All layers frozen except the final fully connected layer  
  - **ResNet-Half**: First half of the layers frozen, second half trainable  
  - **ResNet-Full**: All layers trainable  
- **Output Layer**: Modified to predict 10 classes  

---

## 📁 Dataset

- **Source**: [Sign Language Digits Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)  
- **Structure**: Images organized into folders named `0/`, `1/`, ..., `9/`  
- **Image Format**: RGB  
- **Image Size**: 100×100 pixels (resized depending on the model)  
  - **NN & CNN**: Resized to 64×64  
  - **ResNet**: Resized to 256×256 and center cropped to 224×224  

---

## Key Components

- **Custom Dataset Class**: `SignLanguageDigits` for loading and preprocessing images  
- **Data Loaders**: Used `torch.utils.data.DataLoader` for batching and shuffling  
- **Training & Evaluation**: Custom training loops with test set accuracy evaluation  
- **Transforms**:
  - **NN/CNN**: Resize to 64×64, normalize  
  - **ResNet**: Resize → CenterCrop (224×224) → Normalize (ImageNet mean/std)  

---

## Technologies Used

- Python  
- PyTorch  
- torchvision  
- PIL (Python Imaging Library)  
- matplotlib (for visualization)  

---

## Results

| Model           | Test Accuracy (%) |
|----------------|-------------------|
| Fully Connected NN | 77.51 |
| CNN               | 87.56 |
| ResNet-Frozen     | 88.76 |
| ResNet-Half       | 98.80 |
| ResNet-Full       | 98.80 |


---

## Files

- `Sign_Language_Classification.ipynb`: Main notebook containing:
  - Custom dataset class  
  - Model architectures (FCNN, CNN, ResNet variants)  
  - Training and evaluation code  
- `Results Report.pdf`: Summary of model performances and evaluation   

---
