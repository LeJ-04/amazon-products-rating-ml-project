# Amazon Product Rating Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-green)
![NLTK](https://img.shields.io/badge/NLP-NLTK-yellow)

## Project Overview
**Authors:** SAINT ANDRE Jeffrey & MATAR Gabriel
**Context:** This project aims to predict the star rating of Amazon products based on their price, category, and textual user reviews. By building a pipeline that integrates **Natural Language Processing (NLP)** with **Machine Learning regression**, the project seeks to understand the drivers of customer satisfaction.

## Libraries & Tools

The project utilizes the following Python libraries (as seen in the notebook setup):

* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Text Processing & NLP:**
    * `re`, `string` (Regex and string manipulation)
    * `langdetect` (Filtering non-English reviews)
    * `nltk` (Stopwords, WordNetLemmatizer, VADER SentimentIntensityAnalyzer)
* **Machine Learning (Scikit-Learn):**
    * **Preprocessing:** `StratifiedShuffleSplit`, `StandardScaler`
    * **Models:** `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`
    * **Metrics:** `mean_squared_error`, `mean_absolute_error`, `r2_score`

## Project Pipeline

### 1. Data Cleaning & Standardization
* **Currency Parsing:** A custom function `clean_currency` handles Indian Rupee formatting (removing '₹' and commas) to convert `discounted_price` and `actual_price` into float values.
* **Missing Values:** Imputation of missing `rating_count` values using the **median** strategy to handle skewness.
* **Language Filtering:** Implementation of `langdetect` to identify review languages and strictly filter for **English** reviews (removing errors or short text).
* **Text Consolidation:** merging `review_title` and `review_content` into a single `full_review` feature.

### 2. Exploratory Data Analysis (EDA)
* Univariate analysis of target variables (Ratings).
* Distribution analysis of prices (using Log Scale to handle outliers).

### 3. Feature Engineering & NLP
* **Text Metadata:** Extraction of review length (`review_len`) and word counts (`word_count`).
* **Sentiment Analysis:** Application of **NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)** to compute compound sentiment scores from the reviews.
* **Categorical Encoding:** Processing of the `category` column (likely extracting the main category).

### 4. Model Training & Evaluation
The dataset is split using **Stratified Shuffle Split** to ensure the train/test sets represent the same distribution of ratings. The following regression models are trained and compared:

1.  **Linear Regression** (Baseline)
2.  **Random Forest Regressor** (Ensemble Bagging)
3.  **Gradient Boosting Regressor** (Ensemble Boosting)

Models are evaluated using **RMSE** (Root Mean Squared Error) and **MAE** (Mean Absolute Error).

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/LeJ-04/amazon-products-rating-ml-project.git](https://github.com/LeJ-04/amazon-products-rating-ml-project.git)
    cd amazon-products-rating-ml-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn nltk scikit-learn langdetect
    ```

3.  **Run the Notebook:**
    Open `ProjetML.ipynb` in Jupyter Notebook or Google Colab.
