import stockpred as st

data = st.create_dataset("datasets/BTC")
y_pred, y = st.regression(data, 'linear')
P = st.invest(y_pred, y)

