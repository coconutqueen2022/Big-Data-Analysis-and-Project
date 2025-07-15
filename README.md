## Main functions of Part C 🎯

### 1. Data preprocessing
- Text preprocessing (lowercase conversion, HTML tag removal, punctuation cleaning, etc.)
- Feature engineering (VADER sentiment analysis, TF-IDF features, text statistical features)
- Data balancing (oversampling of minority classes)

### 2. Model training
Three classification algorithms are implemented:
- **Logistic Regression**
- **Random Forest**
- **Multinomial Naive Bayes**

### 3. Hyperparameter optimization
Systematic hyperparameter tuning using GridSearchCV:
- Logistic regression: C value, class weight, solver
- Random Forest: number of trees, maximum depth, number of split samples, class weight
- Naive Bayes: smoothing parameter alpha

### 4. Performance evaluation
- Accuracy, precision, recall, F1 score
- Confusion Matrix Visualization
- Feature Importance Analysis (Random Forest)
