# 🕒 Tell-the-Time Using Convolutional Neural Networks  
*A Deep Learning Pipeline for Predicting Time From Analog Clock Images*

This project explores multiple deep learning architectures for predicting time from analog clock images. Using TensorFlow/Keras, the project evaluates classification, regression, multi-head models, and cyclical (sine–cosine) label encoding to identify the most effective representation for this periodic prediction task.

The final optimized cyclical model achieves a **Common-Sense Error (CSE) of 8.5 minutes**, representing a significant improvement over earlier baselines.

---

## 🎯 Objectives

- Develop a CNN-based pipeline to interpret analog clock images.
- Evaluate how different architectural and labeling choices affect model performance.
- Introduce cyclical encoding (sine–cosine) to handle the periodic nature of time.
- Optimize the final model to achieve sub-10-minute prediction error.

---

## 🧠 Approaches Implemented

### **1. Classification Models**
- Trained softmax-based CNNs with:
  - **12 classes** (coarse hour-level)
  - **24 classes** (half-hour resolution)
  - **720 classes** (minute-level)
- Best accuracy: **47%** (12-class model)
- Observed high boundary errors for minute-level prediction due to discontinuities.

### **2. Regression Model**
- Single-output regression predicting time as a continuous fractional hour.
- MAE: **1.94 hours**
- Struggles with wrap-around (e.g., 11:59 → 12:00).

### **3. Multi-Head Model**
- Separate outputs for hours and minutes.
- Hour MAE: **5.66 hours**
- Minute MAE: **9.79 minutes**
- Improved interpretability but limited performance for hour prediction.

### **4. Cyclical Label Encoding (Sine–Cosine)**
- Encodes time as two continuous outputs:  
  \[
  (\sin(\theta), \cos(\theta))
  \]
- Eliminates discontinuities.
- Achieved a CSE of **24.4 minutes**, a major improvement.

### **5. Optimized Cyclical CNN Model**
- Deep Conv5 architecture with:
  - Data augmentation (rotation, translation, zoom)
  - Dropout and L2 regularization
  - ReduceLROnPlateau + EarlyStopping
- **Final performance: 8.5 minutes CSE (~65% improvement)**

---

## 📊 Key Results

| Model Type          | Metric                     | Performance |
|---------------------|----------------------------|-------------|
| Classification (12) | Accuracy                    | 47%         |
| Classification (720)| CSE                         | 182.9 min   |
| Regression          | CSE                         | 114.2 min   |
| Multi-Head          | CSE                         | 93 min      |
| Cyclical Encoding   | CSE                         | 24.4 min    |
| **Optimized Cyclical Model** | **CSE**            | **8.52 min** |

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy, Matplotlib**
- **Google Colab / GPU (T4)**

---

## 🚀 How to Run

# Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
# Open any notebook locally or in Google Colab

## 🏃‍➡️ Run the below commands for the starters
```bash
python classification_2_1a.py
python reg_2_1d.py
```


