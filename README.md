# 📚 Amazon Book Reviews Sentiment Analysis

> **Advanced Predictive Modeling for E-commerce Sentiment Analysis**  
> *Achieving 99.8% Accuracy with Perfect Minority Class Recall*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

## 🎯 Project Overview

This project implements a comprehensive sentiment analysis system for Amazon book reviews, addressing critical challenges in e-commerce analytics including severe class imbalance (97.5% positive vs 2.5% negative reviews) and sentiment-rating disconnects. Our methodology combines advanced feature engineering with machine learning to predict review sentiment from textual content alone.

### 🌟 Key Achievements

- **99.8% Accuracy** with Random Forest model
- **Perfect Recall** for minority class (1-star reviews)
- **Target Exceeded**: All models achieved F1-score > 0.75 for minority class
- **Novel Insights**: Identified asymmetric sentiment correlation patterns
- **Practical Framework**: Complete, implementable solution for e-commerce platforms

## 📊 Research Methodology

Our approach follows a systematic **five-phase methodology**:


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/coconutqueen2022/Big-Data-Analysis-and-Project.git
cd Big-Data-Analysis-and-Project

# Install required packages
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

### Required Packages

```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
nltk >= 3.6
matplotlib >= 3.4.0
seaborn >= 0.11.0
wordcloud >= 1.8.0
tabulate >= 0.8.9
```

## 📁 Project Structure

```
Big-Data-Analysis-and-Project/
├── 📊 data/
│   └── amazon_books_reviews.csv          # Main dataset (10,000 reviews)
|── 1916290_jian hu_partB.ipynb           # Exploratory analysis & visualizations
├── 1916290_jian hu_partC.ipynb           # Main modeling implementation
├── flowchart.png                         # Methodology flowchart
├── 📖 README.md                          # This file
├── 📋 requirements.txt                   # Python dependencies
└── 🔧 pyproject.toml                     # Project configuration
```

## 🎯 Usage

### Running the Complete Pipeline

```bash
# Part B: Exploratory Analysis & Visualizations
1916290_jian hu_partB.ipynb    

# Part C: Predictive Modeling
1916290_jian hu_partC.ipynb 
```

### Key Scripts

#### 1. Exploratory Analysis (`1916290_jian hu_partB.ipynb`)
- **VADER Sentiment Analysis**: Calculates sentiment scores for all reviews
- **Lexical Analysis**: Word frequency and n-gram analysis
- **Visualizations**: Word clouds, sentiment distributions, bigram comparisons
- **Data Balancing**: Handles class imbalance through upsampling

#### 2. Predictive Modeling (`1916290_jian hu_partC.ipynb`)
- **Feature Engineering**: VADER + TF-IDF + Statistical features
- **Model Training**: Random Forest, Logistic Regression, Naive Bayes
- **Hyperparameter Optimization**: GridSearchCV with cross-validation
- **Performance Evaluation**: Comprehensive metrics and visualizations

## 📊 Results & Performance

### Model Performance Comparison

| Model | Accuracy | Precision (1-star) | Recall (1-star) | F1-Score (1-star) |
|-------|----------|-------------------|-----------------|-------------------|
| **Random Forest** | **0.998** | **0.996** | **1.000** | **0.998** |
| Logistic Regression | 0.996 | 0.991 | 1.000 | 0.996 |
| Multinomial Naive Bayes | 0.990 | 0.993 | 0.986 | 0.990 |

### Key Findings

1. **Asymmetric Sentiment Correlation**: 5-star reviews achieve higher positive sentiment scores than 1-star reviews achieve negative scores
2. **Feature Importance**: VADER sentiment scores dominate feature importance (0.092)
3. **Vocabulary Patterns**: Positive words ("great," "love," "wonderful") provide strong predictive signals
4. **Class Imbalance Solution**: Upsampling successfully addresses severe imbalance without performance loss

### Generated Visualizations

- 📊 **Rating Distribution**: Overall review rating patterns
- 🎭 **Sentiment Analysis**: VADER sentiment score distributions
- ☁️ **Word Clouds**: Most frequent words in 1-star vs 5-star reviews
- 📈 **Bigram Analysis**: Top phrase patterns comparison
- 🎯 **Model Performance**: Comprehensive comparison charts
- 📋 **Confusion Matrices**: Detailed classification results
- 🔍 **Feature Importance**: Top predictive features ranking

## 🔬 Technical Details

### Feature Engineering

**Multi-dimensional Feature Set**:
- **VADER Sentiment Scores**: Compound sentiment polarity
- **TF-IDF Unigrams**: 3,000 most important single words
- **TF-IDF Bigrams**: 2,000 most important word pairs
- **Statistical Features**: Text length, word count, average word length, vocabulary richness

### Text Preprocessing Pipeline

1. **Text Cleaning**: HTML tag removal, lowercase conversion
2. **Tokenization**: NLTK word tokenization
3. **Stopword Removal**: Custom stopwords + NLTK stopwords
4. **Lemmatization**: WordNet lemmatizer
5. **Feature Extraction**: TF-IDF vectorization

### Model Architecture

**Three Classification Algorithms**:
- **Random Forest**: Ensemble method for complex patterns
- **Logistic Regression**: Linear model with interpretability
- **Multinomial Naive Bayes**: Probabilistic text classification

**Hyperparameter Optimization**:
- GridSearchCV with 5-fold stratified cross-validation
- F1-score as primary optimization metric
- Comprehensive parameter grids for each model

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

## 🔬 Technical Details

### Feature Engineering

**Multi-dimensional Feature Set**:
- **VADER Sentiment Scores**: Compound sentiment polarity
- **TF-IDF Unigrams**: 3,000 most important single words
- **TF-IDF Bigrams**: 2,000 most important word pairs
- **Statistical Features**: Text length, word count, average word length, vocabulary richness

### Text Preprocessing Pipeline

1. **Text Cleaning**: HTML tag removal, lowercase conversion
2. **Tokenization**: NLTK word tokenization
3. **Stopword Removal**: Custom stopwords + NLTK stopwords
4. **Lemmatization**: WordNet lemmatizer
5. **Feature Extraction**: TF-IDF vectorization

### Model Architecture

**Three Classification Algorithms**:
- **Random Forest**: Ensemble method for complex patterns
- **Logistic Regression**: Linear model with interpretability
- **Multinomial Naive Bayes**: Probabilistic text classification

**Hyperparameter Optimization**:
- GridSearchCV with 5-fold stratified cross-validation
- F1-score as primary optimization metric
- Comprehensive parameter grids for each model

## 📈 Research Contributions

### Novel Insights
- **Sentiment-Rating Relationship**: First comprehensive study of sentiment expression vs numerical ratings
- **Asymmetric Correlation Pattern**: Identified and validated predictive power
- **Multi-dimensional Features**: Successfully integrated multiple feature types
- **Class Imbalance Solutions**: Demonstrated effective upsampling techniques

### Practical Applications
- **E-commerce Platforms**: Improved review systems and user experience
- **Recommendation Systems**: Sentiment-aware product recommendations
- **Content Moderation**: Automated review quality assessment
- **Customer Insights**: Better understanding of customer satisfaction patterns

## 🎓 Academic Context

This project addresses critical gaps in existing sentiment analysis literature:

| Aspect | Previous Work | Our Improvement |
|--------|---------------|-----------------|
| **Accuracy** | 85-94% | **99.8%** |
| **Class Imbalance** | Struggled with imbalance | **Perfect minority recall** |
| **Feature Engineering** | Single approach | **Multi-dimensional features** |
| **Real-world Applicability** | Limited deployment | **Complete implementation** |

## 🔮 Future Work

### Immediate Actions
1. **Deploy Random Forest model** for e-commerce platforms
2. **Implement feature importance insights** for review interfaces
3. **Extend methodology** to other product categories

### Research Priorities
1. **Domain Adaptation**: Cross-category sentiment analysis
2. **Transfer Learning**: Leverage pre-trained models
3. **Interpretable AI**: Better model explanation techniques
4. **Temporal Analysis**: Evolving sentiment patterns over time

## 📚 Documentation

### Code Documentation
- **Inline Comments**: Extensive code documentation
- **Function Descriptions**: Clear method explanations
- **Error Handling**: Robust exception management
- **Path Management**: Cross-platform compatibility

## 🙏 Acknowledgments

- **Dataset**: McAuley-Lab/Amazon-Reviews-2023
- **Libraries**: scikit-learn, NLTK, pandas, matplotlib, seaborn
- **Research**: Building upon VADER sentiment analysis and TF-IDF techniques
- **Academic Support**: Big Data Analytics course framework
