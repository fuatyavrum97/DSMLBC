import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

#Data Preparation(veriyi anlama ve hazırlama)
pd.set_option('display.max_columns',None) #bütün sütunlar gözüksün
#Adım1
df_ = pd.read_csv("C:\\Users\\Fuat\\PycharmProjects\\DSMLBC9\\Week3\\PrivateSource.csv")
df = df_.copy()

#Adım2
df.head(10) #ilk 10 gözlem
df.columns #değişken isimleri
df.describe().T #betimsel istatistik.
df.isnull().sum() #null değer yok.

#Adım3
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

#Adım 4
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

#Adım 5

df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total": "sum",
                                 "customer_value_total": "sum"})

#Adım6
df.sort_values("customer_value_total",ascending=False).head(10)

#adım7
df.sort_values("order_num_total",ascending=False).head(10)

#Fonksiyonlaştırma ??

#Görev 2

#receny, frequency, monetary

df["last_order_date"].max() # en son alım tarihi
analysis_date = dt.datetime(2021,6,1) #calculation of analysis date (en son alım tarihinin 2 gün sonrası)

rfm = df.groupby("master_id").agg({'last_order_date': lambda last_order_date: (analysis_date - last_order_date.max()).days, #recency
                                    'order_num_total': lambda  order_num_total: order_num_total, #frequency
                                     'customer_value_total': lambda customer_value_total: customer_value_total.sum()}) #monetary

rfm.head()

rfm.columns = ['recency','frequency','monetary']
rfm.head()
rfm.describe().T # 0 olan bir değer yok devam edebiliriz.


rfm["recency_score"] = pd.qcut(rfm['recency'],5,labels = [5,4,3,2,1])
rfm["monetary_score"] = pd.qcut(rfm['monetary'],5,labels=[1,2,3,4,5])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"),5,labels=[1,2,3,4,5])
#rank method amacı: daha fazla sayıda aralığa hep aynı değerler geliyor. kullanmadığımız zaman bu hatayı verdi.
#metodu kullanınca ilk gördüğünü ilk sınıfa ata diyoruz sorun çözülüyor.
rfm

#RF_SCORE
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str)+ rfm['frequency_score'].astype(str))
rfm.describe().T

rfm[rfm["RF_SCORE"]=="55"] #Champions
rfm[rfm["RF_SCORE"]=="11"] #Hibernating

#rfm naming
seg_map = {
        r'[1-2][1-2]':'hibernating',
        r'[1-2][3-4]':'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3,4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map,regex=True)

#görev 5.1
#means of segments
rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg(["mean","count"])

#saving new customers which are champions and loyal_customers

new_df = pd.DataFrame()
#sadık müşteriler (5.2.1) kadın kategorisi ?
new_df["new_customer_id"] = rfm[rfm["segment"] == "champions" & "loyal_customers"].index

#görev 5.2.2
new_df["new_customer_id"] = rfm[rfm["segment"] == "cant_loose" & "new_customers" & "about_to_sleep"].index
