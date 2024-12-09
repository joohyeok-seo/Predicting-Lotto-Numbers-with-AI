# Predicting Lotto Numbers with AI
This project applies various machine learning and deep learning techniques to predict lottery numbers based on historical data. Although lottery outcomes are inherently random, this project explores AI's ability to detect patterns and trends within the data.

---

## Project Overview
This project analyzes and predicts gas prices across Canada using MySQL for data storage and Python for data processing.
The workflow consists of:
  1. **Data Collection**:
     - Historical lottery data is collected via a public API and stored in CSV format for preprocessing and analysis.
    
  2. **Data Preprocessin**:
     - Data cleaning, normalization, and feature engineering are performed.
     - Dimensionality reduction techniques, including PCA, are applied for efficient feature selection.
       
  3. **Model Development**:
     - Multiple machine learning algorithms (e.g., Random Forest, XGBoost, KNN, SVM) and PCA-enhanced approaches.
     - Deep learning models (FCNN, LSTM).
     - Hybrid models combining CNN and LSTM for enhanced pattern recognition.

---

## Algorithms Used
### Machine Learning
  1. **Random Forest**:
     - A robust ensemble method applied to classify and predict lottery numbers.
     - Feature importance analysis was performed to understand predictors' significance.
       
  2. **XGBoost**:
     - A gradient boosting algorithm optimized for high performance.
     - Showed significant potential in handling complex data patterns.
       
  3. **K-Nearest Neighbors (KNN)**:
     - Applied to find patterns based on the proximity of historical data points.
     - Evaluated with different values of K for optimization.
       
  4. **Support Vector Machines (SVM)**:
     - Used to identify the hyperplane that separates lottery patterns effectively.
     - Kernel-based methods were tested to explore non-linear patterns.
       
  5. **Combined PCA + Random Forest**:
     - Principal Component Analysis (PCA) was used for dimensionality reduction before applying Random Forest.
     - Improved computational efficiency and reduced overfitting while maintaining performance.

---

### Deep Learning
  1. **Fully Connected Neural Networks (FCNN)**:
     - A feedforward architecture designed to model non-linear relationships in the data.
     - Features ReLU activation for hidden layers and Softmax for classification.
       
  2. **Long Short-Term Memory (LSTM)**:
     - Designed to capture temporal dependencies in sequential data.
     - Addressed challenges like vanishing gradients in time-series analysis.
       
  3. **Hybrid Model (CNN + LSTM)**:
     - Combines Convolutional Neural Networks (CNN) for spatial pattern extraction with LSTM for modeling sequential dependencies.
     - Achieved the best performance among all tested models.

---

## Key Results
1. Pattern Recognition:
- Models identified trends in historical data but struggled to predict exact outcomes due to randomness.
2. PCA + Random Forest:
- Highlighted key dataset components, improving model efficiency and interpretability.
3. Hybrid Model Performance:
- The CNN + LSTM model demonstrated the highest potential by correctly identifying one number (21) in the 1149th draw.

