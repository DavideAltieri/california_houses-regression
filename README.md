# California Houses Regression

This project is machine learning regression task focused on predicting California house values using the popular **California Housing** dataset from *scikit-learn*.  

## Project Goal

The goal is to build and evaluate regression models necessary to understand median home values ​​based on the features of the dataset.

Exploratory data analysis (EDA) is used in a structured environment to analyze the dataset's aggregate characteristics, including feature distributions and potential skewness.

Multiple regression models are considered, ranging from simple approaches to more complex ones:

- **Linear Regression**
- **Random Forest**
- **XGBoost**

Particular attention is paid to evaluating model performance through cross-validation, ensuring robust and reliable evaluation.

Several evaluation metrics are considered:

- **RMSE** (Root Mean Squared Error)

- **MAE** (Mean Absolute Error)

- **R²** (Coefficient of Determination)

Additional feature engineering techniques are introduced to capture relationships between variables to enhance model performance. The impact of these changes is measured by providing before and after results.

Finally, the best-performing model is further optimized through hyperparameter tuning. The project concludes with an in-depth analysis of the model's behavior to reveal key insights.

## Dataset

This project uses the **California Housing dataset** provided by *scikit-learn*.

The dataset is composed by the following features:

- **MedInc**: Median income of households in a block group (measured in tens of thousands of US dollars).
- **HouseAge**: Median age of houses in the block group.
- **AveRooms**: Average number of rooms per household.
- **AveBedrms**: Average number of bedrooms per household.
- **Popoulation**: Total number of people living in the block group.
- **AveOccup**: Average number of people per household.
- **Latitude**: Geographic latitude of the block group.
- **Longitude**: Geographic longitude of the block group.

And the target variable:

- **MedHouseVal**: Median house value in the block group (measured in hundreds of thousands of US dollars).

## Project Workflow

- Load the dataset

- Run textual EDA

  - head of the dataset

  - info report

  - descriptive statistics

  - skewness

  - correlation matrix

- Generate EDA plots

  - target distribution

  - feature histograms

  - correlation heatmap

- Create engineered features

- Split data into train and test sets

- Train multiple regression models

- Run cross-validation

- Fine-tuning on XGBoost with RandomizedSearchCV

- Evaluate all models on the test set

- Analyze the models’ results and insights with a fair comparison.

*The notebook explains the whole workflow step by step during execution*.

## How to Run

If not already installed, it is necessary to install the libraries:

If the libraries are not installed, run:

```bash
pip install -r requirements.txt
```
To execute the full pipeline, simply run:

```bash
python main.py
```

## Results

The results show that XGBoost trained on the feature-engineered dataset is the best-performing model among those considered. It achieves the lowest RMSE and MAE, together with the highest R² score, both during cross-validation and on the test set.

Hyperparameter optimization provides a further, though limited, improvement, suggesting that the model configuration was already effective before tuning. More generally, the similarity between cross-validation and test-set results indicates that the models generalize well and do not show signs of overfitting.

Additional experiments conducted after removing the IncomePerHousehold feature, which had received a substantially higher importance score than all the other features, show that the overall performance remains very similar. This suggests that, although the feature is highly informative, part of the information it provides is already captured by other variables in the dataset. Therefore, the feature appears to be useful, but not strictly essential for achieving strong predictive performance.





