import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# Title
st.title("Loan Defaul Prediction App")

# breaf introduction to app
st.write("""
This web application (based on a machine learning model) returns the probability that a loan of being defaulted or paid in full.
This could help the bank decide on their credit risk management to avoid losses for the loan default. 
""")
st.write("""
Note: The model was trained with data from the U.S. SBA loan database.
""")
# Import model
#@st.cache()
def get_model():
    with open('LD_xgb_Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

if __name__ == "__main__":

    model = get_model()

    ## Transform binary variables
    def create_binary(content):
        if content =="Yes":
            content = 1
        elif content == "Not":
            content = 0
        elif content == "Active":
            content = 1
        elif content == "Not active":
            content = 0
        return content
    
    #
    st.subheader("Data Required")

    # ----Create inputs----
    # Class for New and Real state variables
    bool_var = ("Yes", "Not")
    
    # Class for reccession variable 
    rece_var = ("Active", "Not active")


    # Categorical vairables
    new = st.selectbox("Is business new or already established?", bool_var)
    realestate = st.selectbox("A loan backed by real estate?", bool_var)
    recession = st.selectbox("Loan State during recession(2007-2009)", rece_var)

    # Numerical vairables 
    naics_code = st.number_input("Industry(NAICS code)", min_value=11, max_value=92)
    no_emp = st.number_input("Number of Business Employees", min_value=0, max_value=9945)
    disb_gross = st.number_input("Amount Disbursed")
    grappv_gross = st.number_input("Gross Amount of Loan Approved by Bank")
    sba_appv = st.number_input("SBA's Guaranteed Amount of Approved Loan")
    term = st.slider("Loan term in months", min_value=0, max_value=527)

    # Prediction button
    prediction = st.button("Predict")

    if prediction:

        data = {
            'NAICS_2DIG': naics_code,
            'Term': term,
            'NoEmp':no_emp,
            'New': create_binary(new),
            'DisbursementGross':disb_gross,
            'GrAppv':grappv_gross,
            'SBA_Appv':sba_appv,
            'Recession':create_binary(recession), # Transform to binary variable
            'RealEstate':create_binary(realestate), # Transform to binary variable
            'portion': (sba_appv/grappv_gross)
        }

        data_df = pd.DataFrame(data, index = [0])

        pred = pd.DataFrame(model.predict_proba(data_df), columns = ["paid in full", "Default"])

        st.subheader("This Loan has a %.2f percent of being defaulted and a %.2f percent being paid in full."
        % (100*(pred.iloc[:,1]), 100*(pred.iloc[:,0]))) 

    

