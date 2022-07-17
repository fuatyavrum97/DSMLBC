import pandas as pd
import datetime as dt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%5f' % x) #float sayıların virgülden sonra kaç basamak olduğu

#Part 1

#Step 1 - read dataset
df_ = pd.read_csv("C:\\Users\\Fuat\\PycharmProjects\\DSMLBC9\\Week3\\PrivateSource.csv")
df = df_.copy()
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head()

#Step 2 - outlier & replace with thresholds
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25) #genel verilen değeri bu
    quartile3 = dataframe[variable].quantile(0.75) #genel verilen değeri bu
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe,variable):
      low_limit, up_limit = outlier_thresholds(dataframe, variable)
      dataframe.loc[(dataframe[variable]>up_limit), variable] = up_limit
      up_limit = round(up_limit)
      dataframe.loc[(dataframe[variable]<low_limit), variable] = low_limit
      low_limit = round(low_limit)

      #dataframe.loc[(dataframe[variable]<low_limit), variable] = low_limit #verisetinde - değer yoksa buna gerek yok.


#step3
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

#step4
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

#step5
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

#Part 2

#Step1

df["last_order_date"].max() # result: 30 May 2021
analysis_date = dt.datetime(2021,6,1)

#Step2
#cltv data structure

#Bu kısmı sor tekrardan!!!!
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7 #weekly
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]')) / 7 #timedelta ???
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()


#Step 3
#BG-BND & Gamma-Gamma & CLTV Calculation

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']) ##??

#expected 3 month purchase
cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly']
                                               )

#expected 6 month purchase
cltv_df["expected_purc_6_month"] = bgf.predict(24,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly']
                                               )

#Gamma Gamma Model

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                               cltv_df['monetary_cltv_avg']).head()

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time = 6,
                                   freq= "W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv

cltv_final = cltv_df.merge(cltv, on= "customer_id", how="left")
cltv_final.sort_values(by="cltv",ascending = False).head(20)

#Segments of CLTV and add to dataset
cltv_final["segment"] = pd.qcut(cltv_final["clv"],4,labels = ["D","C","B","A"])




