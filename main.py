import streamlit as st
import pandas as pd
import pickle

import matplotlib.pyplot as plt


model = pickle.load(open('clf_rf.pkl', 'rb'))
enc = pickle.load(open('encoder.pkl', 'rb'))


st.write("""
# Customer's Signup Prediction App



""")   

                                                                                                               
st.write('we are looking to understand the probability of customers signing up for the delivery club. This would allow the client to mail a more targeted selection of customers, lowering costs, and improving ROI.')


## age input form


st.sidebar.header('Specify the input parameters')
distance_from_store = st.sidebar.number_input(
    label = "01. Enter the customer's distance from store",
    min_value = 0.0,
    max_value = 12.0,
    step = 0.01,
    value = 5.0)


## gender input form

gender = st.sidebar.radio(
    label = "02. Enter the customer's gender",
    options = [1, 0])
    # 1. Male
    # 2. Female

## credit score 
credit_score = st.sidebar.number_input(
    label = "03. Enter the customer's credit score",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.01)

## total Sales
total_sales = st.sidebar.number_input(
    label = "04. Enter the customer's total sales",
    min_value = 0,
    max_value = 2000,
    value = 500)
###
total_items = st.sidebar.number_input(
    label = "05. Enter the the total items purchased by the customer",
    min_value = 0,
    max_value = 600,
    value = 150)

###
transaction_count = st.sidebar.number_input(
    label = "06. Enter the customer's transaction count",
    min_value = 0,
    max_value = 100,
    value = 20)

product_area_count = st.sidebar.number_input(
    label = "07. Enter the product area counts",
    min_value = 1,
    max_value = 5,
    value = 2)

average_basket_value = st.sidebar.number_input(
    label = "08. Enter the average basket value",
    min_value = 0,
    max_value = 200,
    value = 50)




if st.sidebar.button('Submit for Prediction'):
    
    ### store our data in dataframe
    new_df = pd.DataFrame({'distance_from_store': [distance_from_store], 'gender':[gender], 'credit_score':[credit_score], 'total_sales': [total_sales],
                           'total_items':[total_items], 'transaction_count':[transaction_count], 
                           'product_area_count':[product_area_count], 'average_basket_value':[average_basket_value]})
    
    
    st.header('Specified Input parameters')
    st.write(new_df)
    st.write('---')
    y_pred_prob = model.predict_proba(new_df)[0][1]
    
    st.header(f"Based on these customer attributes, our model predicts a purchase probability of {y_pred_prob}.")
    
    
    
    
    st.write('---')
    ## apply model pipeline to the input da
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
  
    ## output prediction
   
    Feature_importance=pd.DataFrame(model.feature_importances_)
    Feature_names=pd.DataFrame(new_df.columns)
    Feature_importance_summary = pd.concat([Feature_importance,Feature_names], axis=1)
    Feature_importance_summary.columns=["Feature_impotance","input_variables"]
    Feature_importance_summary.sort_values(by = "Feature_impotance", inplace= True)
    fig, ax = plt.subplots()
    ax.barh(Feature_importance_summary["input_variables"], Feature_importance_summary["Feature_impotance"], color = 'red')
    ax.set_title("Feature Importance of Random Forest")
    st.pyplot(fig)
        
   
 


