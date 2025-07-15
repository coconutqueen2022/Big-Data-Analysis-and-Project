## Main functions of part B 🎯

### 1. Data loading and basic EDA
- Automatically load Amazon book review dataset
- Display basic information of dataset, missing value statistics
- Display rating distribution

### 2. Sentiment analysis (VADER)
- Use VADER sentiment analyzer to calculate composite sentiment score
- Generate sentiment distribution visualization of 1 star vs 5 star reviews
- Includes box plot and density distribution map

### 3. Vocabulary and N-gram analysis
- **Text preprocessing**: morphological restoration, stop word filtering, custom stop words
- **Word cloud generation**: Generate word clouds for 1 star and 5 star reviews respectively
- **N-gram analysis**: Extract and visualize the most common bigrams

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
