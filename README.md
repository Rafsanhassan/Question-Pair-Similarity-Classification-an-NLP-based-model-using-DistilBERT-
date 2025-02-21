# **Question Pair Similarity Classification**  

This repository contains implementations of multiple models for identifying duplicate questions, ranging from traditional machine learning to deep learning approaches, including a transformer-based method using DistilBERT.  

## **Code 1: Traditional & Deep Learning Model Comparison**  
[Google Colab Link]  
This notebook compares various models for question similarity classification:  
- **Traditional ML Models:** Logistic Regression, SVM (using TF-IDF features)  
- **Deep Learning Models:** Basic ANN, LSTM, GRU, Siamese Network  

### **Key Steps:**  
- **Exploratory Data Analysis (EDA):** Understanding data distribution, missing values, and common patterns.  
- **Data Preprocessing:** Text cleaning, tokenization, TF-IDF vectorization, and sequence padding.  
- **Model Training & Evaluation:** Accuracy, Precision, Recall, F1-Score, and ROC AUC.  

### **Results:**  
- **Best Traditional Model:** Logistic Regression (70% accuracy, 0.7243 ROC AUC).  
- **Best Deep Learning Model:** Siamese Network (highest recall of 0.6658, showing potential for identifying true duplicates).  
- **Challenges:** Limited dataset (10K samples), affecting model generalization.  

---

## **Code 2: DistilBERT-Based Optimized Model**  
[Google Colab Link]  
Building on insights from Code 1, this notebook implements a **transfer learning approach** using DistilBERT for improved question similarity classification.  

### **Key Improvements:**  
- **Pretrained Transformer (DistilBERT):** Extracts rich language features.  
- **Enhanced Architecture:** Fully connected layers with Batch Normalization, Dropout, and GELU activation.  
- **Advanced Training Techniques:**  
  - Focal Loss (to handle class imbalance).  
  - AdamW optimizer with weight decay.  
  - CosineAnnealingWarmRestarts for adaptive learning rates.  
  - Gradient accumulation for better memory efficiency.  

### **Performance:**  
- **Accuracy:** 85.91%  
- **Precision:** 77.00%  
- **Recall:** 88.69%  
- **F1-Score:** 82.43%  
- **ROC AUC:** 93.48%  

### **Conclusion:**  
The DistilBERT-based model significantly outperforms traditional and basic deep learning models, achieving a high recall (88.69%) and a robust ROC AUC of 93.48%, making it highly effective for duplicate question detection.  
