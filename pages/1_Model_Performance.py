import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,roc_auc_score,f1_score
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

st.set_page_config(page_icon=r"C:\Users\ADMIN\OneDrive\Pictures\Screenshots\Screenshot 2026-02-17 113716.png")

df = pd.read_csv(r'C:\Users\ADMIN\Downloads\customer-churn\clean_churn_data.csv')
model = pickle.load(open("churn_pipeline.pkl", "rb"))

x_train = df.drop('Churn',axis=1)
y_train = df['Churn']

y_pred = model.predict(x_train)

st.subheader(" Confusion Matrix")

cm = confusion_matrix(y_train, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)

st.pyplot(fig)

st.subheader('Model metrices')

accuracy = accuracy_score(y_train,y_pred)
precision = precision_score(y_train,y_pred)
recall = recall_score(y_train,y_pred)
f1 = f1_score(y_train,y_pred)
auc =roc_auc_score(y_train,y_pred)


st.metric("Accuracy", f"{round(accuracy*100, 3)}%")
st.metric("Precision", f"{round(precision*100, 3)}%")
st.metric("Recall", f"{round(recall*100, 3)}%")
st.metric("F1 Score",f"{round(f1*100, 3)}%")
st.metric("ROC-AUC", f"{round(auc*100, 3)}%")

st.subheader("📄 Classification Report")

report = classification_report(y_train, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.subheader("📈 ROC Curve")

y_prob = model.predict_proba(x_train)[:,1]
fpr, tpr, _ = roc_curve(y_train, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
st.pyplot(fig)

st.info("""
Insight:
Customers with Support Calls ≥ 5 have a 99% churn rate.
This is the strongest churn indicator.
""")
