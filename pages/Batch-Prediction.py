import pickle
import pandas as pd
import streamlit as st

st.title('Batch-Prediction')
model = pickle.load(open('churn_pipeline.pkl','rb'))

file = st.file_uploader(label='Upload a CSV file',type=['csv'])

if file is not None:

    data = pd.read_csv(file)

    st.subheader('Your dataFrame is:')
    st.dataframe(data)
    st.divider()

    if 'Churn' in data.columns:
        try:
            data = data.drop(columns='Churn',axis=1)
            
            st.subheader('Prediction is below')

            pred = model.predict(data)
            prob = model.predict_proba(data)[:,1]

            data["Churn Prediction"] = pred
            data["Churn Probability"] = prob

            data['Churn Prediction'] = data['Churn Prediction'].map({
                0 : 'Not churn',
                1 : 'Churn'
            })

            st.dataframe(data.head(10))

            st.divider()

            st.subheader('You can download file from Below: ')

            csv = data.to_csv(index=False).encode('utf-8')

            st.download_button(label='Download',data=csv,file_name='churn_prediction.csv',mime="text/csv")

        except Exception as e:
        
            st.error("⚠️ Something went wrong during prediction.")
            st.write("Error Details:", e)