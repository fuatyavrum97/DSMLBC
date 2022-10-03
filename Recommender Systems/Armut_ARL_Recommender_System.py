import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#çıktının tek bir satırda olmasını istiyorsak
pd.set_option('display.expand_frame_repr', False)

#Rule1

#Step 1
df_ = pd.read_csv("Week5/datasets/armut_data.csv")
df = df_.copy()
df.head()

#Step 2
df["Hizmet"] = df["ServiceId"].apply(str) + '_' + df["CategoryId"].apply(str)  # concat
df["Hizmet"].head()

#Step 3
df.head()

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]