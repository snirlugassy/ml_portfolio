import pandas as pd
import numpy as np
from sklearn.linear_model import LassoLars
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class Regression:


    def download_dataset(start_date, end_date):
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_tickers = wiki_table[0]
        tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
        data = yf.download(tickers, start_date, end_date)
        return data


    def predLars(train_data:pd.DataFrame): 

        # Convert the index to datetime if it's not already
        # train_data.index = pd.to_datetime(train_data.index)
        train_data_filled = train_data.fillna(train_data.mean())
        prediction_horizon = 1
        returns = train_data_filled.pct_change()
        returns = returns.fillna(train_data.mean())
        last_predicted_values = {}
        true_values = {}
        mses = {}
        # Loop through each column and perform predictions
        for column in returns.columns:
            target_data = returns[column].values
            # Create lagged features for time series prediction
            X = []
            y = []
            for i in range(len(target_data) - prediction_horizon):
                X.append(target_data[i:i + prediction_horizon])
                y.append(target_data[i + prediction_horizon])
            X = np.array(X)
            y = np.array(y)
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
            
            # Initialize and train the LassoLars model
            model = LassoLars(alpha=0.01)
            model.fit(X_train, y_train)
            
            # Predict future costs
            y_pred = model.predict(X_test)
            last_predicted_values[column] = model.predict([X_test[-1]])[0]
            true_values[column] = y_test[-1]
            # Evaluate the model's performance
            mse = mean_squared_error(y_test, y_pred)
            mses[column] = mse

        top_columns = sorted(last_predicted_values, key=last_predicted_values.get, reverse=True)[:10]
        mean_returns_highest = [np.mean(last_predicted_values[col]) for col in top_columns]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot for top 10 highest columns
        ax1.bar(top_columns, mean_returns_highest, label='Mean Predicted Return')
        ax1.bar(top_columns, [np.mean(true_values[col]) for col in top_columns], label='Mean True Return', alpha=0.5)
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Mean Return')
        ax1.set_title('Mean Predicted vs. True Returns for Top 10 Highest Mean Return Columns')
        ax1.set_xticks(range(len(top_columns)))
        ax1.set_xticklabels(top_columns, rotation=45, ha='right')
        ax1.legend()

        top_lowest_mse = sorted(mses, key=mses.get)[:10]
        mean_mses_lowest = [mses[col].mean() for col in top_lowest_mse]
        # Plot for top 10 lowest MSE columns
        ax2.bar(top_lowest_mse, mean_mses_lowest, color='green', label='Mean MSE')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('Top 10 Lowest Mean Squared Error Columns')
        ax2.set_xticks(range(len(top_lowest_mse)))
        ax2.set_xticklabels(top_lowest_mse, rotation=45, ha='right')
        ax2.legend()

        plt.tight_layout()
        plt.show()


    if __name__ == "__main__":
        START_DATE = '2018-07-01'
        END_TRAIN_DATE = '2023-05-31'
        END_TEST_DATE = '2023-06-30'
        data1 = download_dataset('2023-04-01', END_TRAIN_DATE)
        data_train = data1['Adj Close']
        # print(data_train)
        print(predLars(data_train))

