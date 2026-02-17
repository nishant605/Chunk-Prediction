import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("churn_pipeline.pkl", "rb"))

st.title('Customer churn prediction')
st.write('Enter details below:')

st.divider()

tenure_ip = st.number_input('Tenure(Months)',min_value=0,max_value=50)
freq_ip = st.number_input('Usage Frequency',min_value=0,max_value=50)
calls_ip = st.number_input('Support Calls',min_value=0,max_value=50)
payment_ip = st.number_input('Payment Delay (days)',min_value=0,max_value=50)
interaction_ip = st.number_input('Last Interaction (days ago)',min_value=0,max_value=1000)
spend_ip = st.number_input('Total Spend',min_value=0,max_value=1000000000)

subs_ip = st.selectbox('Choose Subscription type',['Standard', 'Premium', 'Basic'])

sub_standard = 0
sub_premium = 0
sub_Basic = 0

if subs_ip == 'Standard':
    sub_standard = 1
elif subs_ip == 'Premium':
    sub_premium = 1
else:
    sub_Basic = 1

new_customer = pd.DataFrame([{
    "Tenure": tenure_ip,
    "Usage Frequency": freq_ip,
    "Support Calls": calls_ip,
    "Payment Delay": payment_ip,
    "Last Interaction": interaction_ip,
    "Total Spend": spend_ip,
    "Subscription Type_Standard": sub_standard,
    "Subscription Type_Premium": sub_premium,
    "Subscription Type_Basic": sub_Basic
}])

btn_ip = st.button(label='predict')
st.divider()

if btn_ip:
    pred = model.predict(new_customer)
    pred = model.predict(new_customer)[0]
    prob = model.predict_proba(new_customer)[0][1]

    st.subheader("📌 Prediction Result")

    if pred == 1:
        st.error("❌ Customer Will CHURN")
    else:
        st.success("✅ Customer Will NOT Churn")

    st.write(f"### Churn Probability: **{prob:.2f}**")

    # Risk Level
    if prob > 0.8:
        st.warning("🚨 Very High Risk Customer")
    elif prob > 0.6:
        st.warning("🔥 High Risk Customer")
    elif prob > 0.3:
        st.info("⚠️ Medium Risk Customer")
    else:
        st.success("✅ Low Risk Customer")