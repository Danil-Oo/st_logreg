import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.title('Логарифмическая регрессия')
st.divider()

st.subheader('Вот сюда можно данные закинуть')
test = st.file_uploader('**Загурзите данные в формате csv**', type='csv')

if test is not None:
    test = pd.read_csv(test)
    st.write(test.head(5))
else:
    test = pd.read_csv('../aux/credit_train.csv')

st.divider()

train = pd.read_csv('../aux/credit_train.csv')

# Нормируем данные
ss = StandardScaler()
X, y = ss.fit_transform(train[['CCAvg', 'Income']]), train['Personal.Loan']
x_test, y_test = ss.transform(test[['CCAvg', 'Income']]), test['Personal.Loan']

# Пишем класс


class LogReg:
    def __init__(self, learning_rate, n_inputs, iterations=40):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = np.random.uniform(size=n_inputs)
        self.intercept_ = np.random.uniform()
        self.iterations = iterations

    def sigmoida(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        for i in range(self.iterations):
            y_pred = self.sigmoida(self.intercept_ + X@self.coef_)
            derivative_w0 = (y_pred - y).mean()
            derivative_w1 = X.T@(y_pred - y) / X.shape[0]
            w0_new = self.intercept_ - self.learning_rate * derivative_w0
            w1_new = self.coef_ - self.learning_rate * derivative_w1
            self.intercept_ = w0_new
            self.coef_ = w1_new
        return self.coef_, self.intercept_

    def predict(self, X):
        return self.sigmoida(self.intercept_ + X@self.coef_)


testing_logreg = LogReg(0.1, 2, 1000)
lr_line = testing_logreg.fit(X, y)


st.subheader('Вот такие вот коэффициентики получились у модели после обучения')
col1, col2, col3 = st.columns(3)
col1.metric("CCAvg", round(lr_line[0][0], 4))
col2.metric("Income", round(lr_line[0][1], 4))
col3.metric("Personal.Loan", round(lr_line[1], 4))
st.divider()

st.subheader('Вот тут можно глянуть твой результат регрессии')
df = pd.DataFrame(testing_logreg.predict(x_test),
                  columns=['Вероятности получения кредита'])


def highlight_low_values(val):
    color = 'background-color: #ff9999' if val < 0.5 else ''
    return color


st.dataframe(df.style.applymap(highlight_low_values), hide_index=True)
st.divider()

st.subheader('Вот тут вот графичек твоей регрессии')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu)
x_ax = np.linspace(x_test[:, 0].min(), x_test[:, 0].max(), 100)
y_ax = - (lr_line[0][0] / lr_line[0][1]) * x_ax - (lr_line[1] / lr_line[0][1])
plt.plot(x_ax, y_ax, c='black', label='Разделяющая прямая', linewidth=3)
plt.xlabel('CCAge')
plt.ylabel('Income')
plt.legend()
legend_labels = ['Одобрено', 'Отказано']
colors = ['blue', 'red']
for color, label in zip(colors, legend_labels):
    plt.scatter([], [], c=color, label=label)

plt.legend(loc='lower right')
st.pyplot(plt)
