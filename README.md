Predictive Model for Housing Prices
This project demonstrates how to build a predictive model to forecast housing prices using the California Housing Dataset. The model is built using Linear Regression and evaluated using various performance metrics. The goal is to showcase the ability to analyze data, preprocess it, and train a model using scikit-learn.

Project Overview
In this project, we use the California Housing Dataset, which contains information about various features such as the average number of rooms, population, and the median income of areas in California. The task is to predict the median house value (PRICE) for each block group based on these features.

The steps covered in this project are:

Loading the dataset.
Data exploration and visualization.
Data preprocessing (splitting into training and testing sets).
Building a Linear Regression model.
Evaluating the model performance using metrics like MAE, MSE, RMSE, and R-squared.
Saving and loading the trained model for future use.
Key Skills Used
Data Preprocessing
Model Building with scikit-learn
Model Evaluation
Data Visualization with matplotlib and seaborn
Saving and Loading models using joblib
Installation
To run this project, you need to have Python installed on your machine along with the required libraries. You can install the required libraries using pip:


pip install scikit-learn pandas matplotlib seaborn joblib numpy
Project Structure
machine.py: Python script for training the model and making predictions.
house_price_model.pkl: The saved model after training (optional for loading and predicting with an existing model).
README.md: Project documentation.
Usage
Clone the repository:
git clone https://github.com/Spandana928/house-price-predictor.git
cd house-price-predictor
Run the Python script:

python machine.py
You can also load the saved model (house_price_model.pkl) for prediction:

python
model = joblib.load('house_price_model.pkl')
predictions = model.predict(X_test)
Evaluation Metrics
The following evaluation metrics are used to assess the model’s performance:

Mean Absolute Error (MAE): The average of the absolute errors between actual and predicted values.
Mean Squared Error (MSE): The average of the squared errors.
Root Mean Squared Error (RMSE): The square root of the MSE.
R-squared (R²): The proportion of the variance in the dependent variable that is predictable from the independent variables.
Example Results
After training the model, the following results are obtained:

MAE: X.XX
MSE: X.XX
RMSE: X.XX
R²: X.XX
A scatter plot comparing actual vs. predicted prices is also shown for better visualization of the model's performance.

Contributions
Feel free to fork the repository, contribute improvements, or report issues. Contributions are always welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.
