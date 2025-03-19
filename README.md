# Tomato-Disease-Classification-Using-CNN
# 🍅 Tomato Leaf Disease Classification using CNN  

## 📌 Project Overview  
This project uses **Convolutional Neural Networks (CNNs)** to classify **11 different types of tomato leaf diseases**. The model is trained on an augmented dataset downloaded from **Kaggle** and evaluates images for accurate disease detection.  

---

## 📂 Dataset  
- **Source:** (https://www.kaggle.com/datasets/ashishmotwani/tomato)
- **Classes:** 11 categories of tomato leaf diseases  
- **Preprocessing:** Rescaled pixel values (1./255), Data Augmentation  

---

## 🛠 Tech Stack  
- **Deep Learning Framework:** TensorFlow/Keras  
- **Programming Language:** Python  
- **Dataset Processing:** ImageDataGenerator  
- **Evaluation Metrics:** Accuracy, Loss, Confusion Matrix  

---

## 🚀 Model Architecture  
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(11, activation='softmax')  # 11-class classification
])
