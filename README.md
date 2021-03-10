# Stock-Prediction

The purpose of this project is to help investors invest in the financial market. Furthermore, this software allows the user to add any dataset that contains the price information of a financial asset over a period of time. The software then utlizes a ML algorithm to predict future prices and implements a traiding strategy to show much interest can be made from buying athat particual financial market.

The procedure tof how to predict and test trading strategies is the following.

##.  Add a dataset(.csv file)containting the "open", "high", "low" and "close" price of any financial
    asset. 

##  Using the open, high, low and close prices, we compute other indicator variables

## Initiate the machine learning step. We use 80% of the observations to train our model and we the
evaluate the trained model on the rest of the data. the ope, high, low prices, together with the
indicator variables are used to predict the future closing prices. You can choose to use the
following machine learning algorithms

### Linear Regression
### Support Vector Machine
### Nearest Neighbor
### Decision Tree

The performance of the algorithms are evaluated using the

### Mean Average Procentage Error
### Pearson correlation
### Theil U
