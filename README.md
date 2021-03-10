# Stock-Prediction

The purpose of this project is to help investors invest in the financial market. Furthermore, this software allows the user to add any dataset that contains the price information of a financial asset over a period of time. The software then utlizes a ML algorithm to predict future prices and implements a traiding strategy to show much interest can be made from buying athat particual financial market.

The procedure tof how to predict and test trading strategies is the following.

1.  Add a dataset(.csv file)containting the "open", "high", "low" and "close" price of any financial
    asset. 

2.  Using the open, high, low and close prices, we compute other indicator variables

3. Initiate the machine learning step. We use 80% of the observations to train our model and we the
evaluate the trained model on the rest of the data. the ope, high, low prices, together with the
indicator variables are used to predict the future closing prices. You can choose to use the
following machine learning algorithms

a) Linear Regression
b) Support Vector Machine
c) Nearest Neighbor
d) Decision Tree

The performance of the algorithms are evaluated using the

1.) Mean Average Procentage Error
2.) Pearson correlation
3.) Theil 

4. Implement the following strategy!
    if y(t+1) > y(t): Buy
    if y/t+1) < y(t): Sell
