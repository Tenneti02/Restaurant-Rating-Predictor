# Restaurant-Rating-Predictor
Predict restaurant ratings using ML! 🍽️📊 Built with Random Forest and advanced data preprocessing.
# 🍽️ Restaurant Rating Prediction 🚀

## 🌟 Overview
Have you ever wondered how restaurants are rated? This project predicts a restaurant's aggregate rating based on various features such as price range, availability of table booking, online delivery, cuisines, and location. ** Random Forest Regression ** powers the model with advanced data preprocessing techniques like one-hot encoding, standardization, and imputation.

## 📊 Dataset
The dataset is loaded from an **Excel file (`Dataset.xlsx`)**, and the following features play a crucial role in the prediction:

- **💰 Price range** (Numeric)
- **📅 Has Table booking** (Categorical: Yes/No converted to 1/0)
- **🚀 Has Online delivery** (Categorical: Yes/No converted to 1/0)
- **🍽️ Cuisines** (Categorical)
- **💵 Average Cost for two** (Numeric)
- **🌍 City** (Categorical)

The target variable is **Aggregate Rating**, which the model learns to predict.

## 🛠️ Model Pipeline
The pipeline consists of:
1. **🔄 Data Preprocessing**
   - Numerical features are imputed (median strategy) and standardized.
   - Categorical features are imputed (constant value) and one-hot encoded.
2. **🤖 Model Training**
   - A **Random Forest Regressor** is used with 100 estimators and a fixed random state for reproducibility.

## 🚀 Steps to Run the Project
1. Install necessary dependencies:
   ```bash
   pip install pandas scikit-learn openpyxl
   ```
2. Load the dataset (ensure `Dataset.xlsx` is in the same directory or update the path accordingly).
3. Run the script to train the model and evaluate it.
4. The model will output:
   - **📉 Mean Squared Error (MSE)**
   - **📈 R-squared Score (R²)**
5. Predict the rating for a new restaurant using predefined sample data.

## 📊 Model Evaluation
- The model evaluates performance using **Mean Squared Error (MSE)** and **R-squared Score (R²)**.
- The results will be printed in the console, giving insights into how well the model performs.

## 🍕 Example Prediction
Imagine a new restaurant opening in **New York**! The model can predict the **aggregate rating** based on given inputs:
```python
new_restaurant = pd.DataFrame({
    'Price range': [3],
    'Has Table booking': [1],
    'Has Online delivery': [0],
    'Cuisines': ['Italian, Mediterranean'],
    'Average Cost for two': [2000],
    'City': ['New York']
})
```
The predicted rating will be printed after processing through the pipeline. 📊

## 🔮 Future Improvements
- Fine-tuning hyperparameters for better performance.
- Exploring other regression models.
- Handling missing values and categorical encoding more effectively.
- Adding more features for better accuracy.

## 📂 Repository Structure
```
├── Dataset.xlsx  # Input dataset
├── model.py      # Main script for training and prediction
├── README.md     # Project documentation
```

## ✍️ Author
[Your Name]

## 📜 License
This project is open-source under the MIT License. Feel free to explore, modify, and contribute! 🚀

