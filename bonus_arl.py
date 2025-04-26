import pandas as pd
from xarray.util.generate_ops import inplace

pd.set_option("display.max_columns", None)  # Tüm sütunları göster
pd.set_option("display.width", 1000)  # Satır genişliğini artır
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("C:/Users/merve/OneDrive/Desktop/miuul/Assocaiton_Recommender/online_retail_II.xlsx",sheet_name="Year 2010-2011")

df = df_.copy()


def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 -quartile1
    up_limit = quartile3 + interquantile_range * 1.5
    low_limit = quartile1 - interquantile_range * 1.5

    return up_limit,low_limit


def replace_with_thresholds(dataframe,variable):
    up_limit, low_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_dataframe_prep(dataframe):
    dataframe.dropna(inplace = True)
    #dataframe = dataframe[~dataframe["Invoice"].str.contains("C",na = False)]
    dataframe = dataframe[~dataframe["StockCode"].str.contains("POST",na = False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]

    replace_with_thresholds(dataframe,"Quantity")
    replace_with_thresholds(dataframe,"Price")

    return dataframe


df = retail_dataframe_prep(df)


create_invoice_product = df.groupby(["Invoice","Description"]).agg({"Quantity" : "sum"}).\
    unstack().\
    fillna(0).\
    map(lambda x : 1 if x > 0 else 0).head(20)

# Function to create binary purchase matrix
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0).\
        map(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).\
        map(lambda x: 1 if x > 0 else 0)

# Example usage:
# Using Description
invoice_product_matrix = create_invoice_product_df(df, id=False)

# Using StockCode
invoice_product_matrix_by_id = create_invoice_product_df(df, id=True)

# To view first 5 rows and 5 columns of the result:
print(invoice_product_matrix.iloc[0:5, 0:5])


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


invoice_product_matrix_by_id = invoice_product_matrix_by_id.astype(bool)
print(type(invoice_product_matrix_by_id))



def create_rules(dataframe,id =True):
    if dataframe.empty:
        raise ValueError("Dataframe is empty")
    else:
     invoice_product_matrix_by_id = create_invoice_product_df(dataframe,id)
     invoice_product_matrix_by_id = invoice_product_matrix_by_id.astype(bool)

     frequent_items = apriori(invoice_product_matrix_by_id,
                              min_support=0.01,
                              use_colnames=True)

     frequent_items.sort_values("support", ascending=False)

     rules = association_rules(frequent_items,
                               metric="confidence",
                               min_threshold=0.01)

     return rules


rules = create_rules(df)

rules = pd.DataFrame(rules)


def arl_recommender(rules_df,product_id,rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 3)