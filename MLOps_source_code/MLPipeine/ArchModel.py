from arch import arch_model
from scipy.stats.distributions import chi2
import pickle

class Arch_Model:

    def __init__(self, df_train, df_test):

        self.lower_order_model(df_train)

        # ## Higher-Lag ARCH Models
        self.higher_order_model(df_train)

        # ## GARCH Model
        self.garch(df_train)

        self.forecasting(df_train, df_test)


    def garch(self, df_train):
        model_garch_1_1 = arch_model(df_train.returns[1:], mean="Constant", vol="GARCH", p=1, q=1, dist="Normal")
        results_garch_1_1 = model_garch_1_1.fit(update_freq=5)
        pickle.dump(results_garch_1_1, open("../Output/model.pkl", "wb"))
        results_garch_1_1.summary()

    def higher_order_model(self, df_train):
        model_arch_2 = arch_model(df_train.returns[1:], mean="Constant", vol="ARCH", p=2, dist="Normal")
        results_arch_2 = model_arch_2.fit(update_freq=5)
        results_arch_2.summary()
        model_arch_3 = arch_model(df_train.returns[1:], mean="AR", vol="ARCH", p=3, dist="Normal")
        results_arch_3 = model_arch_3.fit(update_freq=5)
        results_arch_3.summary()

    def lower_order_model(self, df_train):
        model_arch_1 = arch_model(df_train.returns[1:])
        results_arch_1 = model_arch_1.fit(update_freq=5)
        results_arch_1.summary()
        model_arch_1 = arch_model(df_train.returns[1:], mean="Constant", vol="ARCH", p=1, dist="Normal")
        results_arch_1 = model_arch_1.fit(update_freq=5)
        results_arch_1.summary()
        model_arch_1 = arch_model(df_train.returns[1:], mean="Constant", lags=[4, 3, 6], vol="ARCH", p=1, dist="normal")
        results_arch_1 = model_arch_1.fit(update_freq=5)
        results_arch_1.summary()

    def LLR_test(self, mod_1, mod_2, DF=1):
        L1 = mod_1.fit(start_ar_lags=11).llf
        L2 = mod_2.fit(start_ar_lags=11).llf
        LR = (2 * (L2 - L1))
        p = chi2.sf(LR, DF).round(3)
        return p

    # ## The arch_model() Method
    def forecasting(self, df_train,df_test):
        # ## Forecasting the Results
        start_date = "2019-03-31"
        df_test["returns"] = df_test.Banking.pct_change(1) * 100
        mod_arch = arch_model(df_train.returns[1:], mean="Constant", vol="ARCH", p=2, dist="Normal")
        res_arch = mod_arch.fit(last_obs=start_date, update_freq=10)
        res_arch.summary()
        pred = res_arch.forecast(horizon=10)
        print(pred.residual_variance)




