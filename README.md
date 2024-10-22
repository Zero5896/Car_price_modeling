# Car Price Modeling and Analysis

## Description
This project is about testing my Data Science and Machine Learning skills through a Kaggle dataframe [insert link]. The main goal is to answer questions posed on Kaggle and assess how good the dataframe is and how usable it can be for users.

## Dataset Description
The dataset, obtained from Kaggle, contains information on various car features, including:
- **Brand**: The manufacturer of the car
- **Year**: The year the car was manufactured
- **Engine Size**: The size of the engine in liters
- **Fuel Type**: The type of fuel used by the car (e.g., Petrol, Diesel, Electric, Hybrid)
- **Transmission**: The type of transmission (e.g., Manual, Automatic)
- **Mileage**: The distance the car has traveled
- **Condition**: The condition of the car (e.g., New, Used, Like New)
- **Price**: The market price of the car
- **Model**: The specific model of the car

## Technologies Used
- Python
- Pandas
- Scikit-learn (Linear Regression, Decision Trees, Random Forests, Neural Networks)
- Matplotlib
- Seaborn

## Installation Instructions
To run this project, ensure you have Python installed. You can set up your environment and install the necessary packages using the following command:

```bash
pip install -r requirements.txt
```

## Results

### pending

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```
# Split the data
X = data.drop(columns=['Price'])
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate a model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE of Linear Regression: {rmse}')
