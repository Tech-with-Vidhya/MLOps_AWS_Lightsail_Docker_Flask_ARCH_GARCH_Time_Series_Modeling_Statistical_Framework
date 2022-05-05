# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from MLPipeine.utils import read_data
from MLPipeine.ArchModel import Arch_Model



# importing the data
raw_csv_data = read_data("../Input/CallCenterData.xlsx")

# check point of data
df_comp = raw_csv_data.copy()

# ## Setting date as Index
# taken as a date time field

df_comp.set_index("month", inplace=True)

# seeting the frequency as monthly
df_comp = df_comp.asfreq('M')

# checking for the null values

# ## Time Series Visualization
df_comp.Healthcare.plot(figsize=(20,5), title="Healthcare")
plt.savefig("../Output/"+"health.png")


df_comp.Telecom.plot(figsize=(20,5), title="Telecom")
plt.savefig("../Output/"+"telecome.png")


df_comp.Banking.plot(figsize=(20,5), title="Banking")
plt.savefig("../Output/"+"Banking.png")


df_comp.Technology.plot(figsize=(20,5), title="Technology")
plt.savefig("../Output/"+"Technology.png")


df_comp.Insurance.plot(figsize=(20,5), title="Insurance")
plt.savefig("../Output/"+"Insurance.png")


# ## Train Test Split

import warnings
warnings.filterwarnings("ignore")


# train set split
test_size = 22
df_train = df_comp[:-test_size]
df_test = df_comp[-test_size:]


# ## Calculating Returns and Volatility

df_train['returns'] = df_train.Banking.pct_change(1)*100

df_train['sq_returns'] = df_train.returns.mul(df_train.returns)
'''
df_train.returns.plot(figsize=(20,5))
plt.title("Returns", size = 24)
plt.savefig("../Output/"+"returns.png")


df_train.sq_returns.plot(figsize=(20,5))
plt.title("Volatility", size = 24)
plt.savefig("../Output/"+"Volatility.png")


sgt.plot_pacf(df_train.returns[1:], lags = 40, alpha = 0.05, zero = False , method = ('ols'))
plt.title("PACF of Returns", size = 20)
plt.savefig("../Output/"+"PACFReturns.png")


sgt.plot_pacf(df_train.sq_returns[1:], lags = 40, alpha = 0.05, zero = False , method = ('ols'))
plt.title("PACF of Returns", size = 20)
plt.savefig("../Output/"+"ACFReturns.png")
'''

# train the model and derive the inferences
Arch_Model(df_train, df_test)