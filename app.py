from flask import Flask, render_template, request, jsonify, url_for
import yfinance as yf
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np

# Ensure that the static folder exists
static_folder = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_form')
def predict_form():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the required fields are in the request
    if 'symbol' not in request.form or 'start_date' not in request.form or 'end_date' not in request.form or 'model' not in request.form or 'future_days' not in request.form:
        return "Missing form data. Please ensure all fields are filled out.", 400
    
    symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    model_type = request.form['model']
    future_days = int(request.form['future_days'])
    
    # Fetch historical stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    # Feature Engineering
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()
    
    # Drop rows with missing values
    stock_data.dropna(inplace=True)
    
    # Check if there are enough samples remaining
    if len(stock_data) < 2:
        return "Not enough data remaining after dropping rows with missing values.", 400
    
    # Extract features and target variable
    X = stock_data[['Open', 'High', 'Low', 'Volume', 'MA_50', 'MA_200']]
    y = stock_data['Close']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model selection and training
    if model_type == 'ridge':
        parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}  # Regularization parameter
        model = Ridge()
        grid_search = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5, error_score='raise')
        grid_search.fit(X_train, y_train)
        best_alpha = grid_search.best_params_['alpha']
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
    elif model_type == 'linear':
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif model_type == 'tree':
        parameters = {'max_depth': [5, 10, 15, 20, None]}  # Example hyperparameters
        model = DecisionTreeRegressor(random_state=42)
        grid_search = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        model.fit(X_train, y_train)
    else:
        return "Invalid model type selected.", 400
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    plt.figure(figsize=(18, 10))  
    
    # Plot actual prices as a line with markers
    plt.plot(stock_data.index, stock_data['Close'], label='Actual Prices', color='blue', alpha=0.7, linestyle='-', marker='o')
    
    # Plot predicted prices as a line with markers
    plt.scatter(X_test.index, predictions, label='Predicted Prices', color='green', marker='o')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Market Prices')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Save the plot to a file
    plot_path = os.path.join('static', 'plot.png')
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        return f"Error saving plot: {e}", 500
    
    # Get the last predicted value
    last_predicted_value = predictions[-1]
    
    # Convert R-squared to percentage
    r2_percentage = r2 * 100
    
    # Future prediction
    future_dates = [datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=i) for i in range(1, future_days + 1)]
    future_predictions = []

    last_row = X.iloc[-1].values.reshape(1, -1)
    
    for _ in future_dates:
        next_pred = model.predict(last_row)
        future_predictions.append(next_pred[0])
        
        # Update features for the next day prediction
        last_row = np.copy(last_row)
        last_row[0, 0] = next_pred[0]  # Update 'Open' with the predicted 'Close'
        
        
    future_predictions = [round(pred, 2) for pred in future_predictions]

    # Pass the predicted values, evaluation metrics, dataset sample, and future predictions to the template
    return render_template('result.html', symbol=symbol, actual_prices=y_test.values, predicted_prices=predictions, mse=mse, mae=mae, r2=r2_percentage, plot_path=url_for('static', filename='plot.png'), last_predicted_value=last_predicted_value, stock_data=stock_data.head(), future_dates=[date.strftime('%Y-%m-%d') for date in future_dates], future_predictions=future_predictions, zip=zip)

if __name__ == '__main__':
    app.run(debug=True)