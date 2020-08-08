"""Streamlit mockup of EHR system"""

import streamlit as st
import pandas as pd

patnum = 2 #Number of patients to include
df = pd.read_csv("training_v2.csv")

st.sidebar.title(f"ICU EHR System")
st.sidebar.markdown(f"Currently {patnum} beds in use")
st.sidebar.markdown("## View patient records")
st.markdown("### Patient record")
id = st.sidebar.selectbox("Patient ID",range(patnum))
df = df.drop(["patient_id", "hospital_id", "hospital_death"],axis=1)
st.dataframe(df.iloc[id])
st.markdown(f"[MortalityPredict (SMART)](https://smart.encqui.re?server=https://fhir.encqui.re&id={id}&token=544546184779123)")
