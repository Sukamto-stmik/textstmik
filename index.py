import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import cv2
import pandas as pd
#from st_aggrid import AgGrid
#import plotly.express as px
import io 
import pandas as pd
from io import StringIO
import sys
import csv
import psycopg2
import subprocess
import time
import os
from os.path import exists
import pathlib
from pathlib import Path

# koneksi to database
@st.cache_resource
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])
conn = init_connection()

@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

## List Menu    
with st.sidebar:
    choose = option_menu("Project Menu", ["", "About", "Upload File", "Proses TextMining", "Report", "Contact"],
                         icons=['', 'house', 'cloud', 'kanban', 'newspaper','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "About":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">About</p>', unsafe_allow_html=True)
        st.write('')    
        st.subheader(f'Kelompok  : \n Sukamto (22575002) \n \n Arief Hidayat \n\n')
    with col2:               # To display brand log
        st.write('')
        #st.image(logo, width=130 )

## Menu untuk upload file data source
elif choose == "Upload File":
        # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Upload File</p>', unsafe_allow_html=True)
        st.write('')    
    
        # To display brand log
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            #st.write(bytes_data)
            
            dataframe = pd.read_csv(uploaded_file, encoding='latin-1')
            st.write(dataframe)
            btInput = st.button("upload file")
            if btInput:
                dt = pd.DataFrame(dataframe)
                dt.to_csv('D:\\Learning\\python\\project\\data_input\\data_input.csv', index=True, encoding='latin-1')

                ## memanggil file insert_data.py
                subprocess.run([f"{sys.executable}", "insert_data.py"])
                st.success(f'            >>>  upload file success.')

## Menu proses
elif choose == "Proses TextMining":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">SMS SPAM Klasifikasi </p>', unsafe_allow_html=True)

    ## Menjalankan proses Text Mining
    showtable = st.button("Start Proceess")
    if showtable:
        ## Menjalankan file model_textmining
        subprocess.run([f"{sys.executable}", "model_textmining.py"])
        progress_text = "Operation in progress. Please wait."
        finish_text = "Operation complete."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=finish_text)
        st.success('    >> Proses Selesai')

# Menu Data Report
elif choose == "Report":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Report Proses</p>', unsafe_allow_html=True)
    st.subheader('-----------------------')

    #Menampilkan data source
    df=pd.read_csv(r'D:\Learning\python\project\data_input\data_input.csv', encoding='latin-1')
    df_head=df.head()
    if st.button('View Data Source', key='1'):
        st.dataframe(data=df)
        
    else:
        st.write('---')

    #Menampilkan data hasil proses Klasifikasi
    showtable = st.button("view Klasifikasi Result")
    if showtable:
        st.write('Klasifikasi Result (1=ham, 0=spam)')
        rsl_kasifikasi = run_query("""
            select a.v1 as id,b.v1 as ham,b.v2 as spam, a.v2 as message  
                from tbl_klasifikasi_text a 
                join tbl_klasifikasi b on b.id=a.v1
            """
        )
        st.write(pd.DataFrame(data=rsl_kasifikasi))  

    #Menampilkan hasil model text Mining
    vReport = st.button("vReport Model")
    if vReport:
        data = pd.read_csv("D://Learning//python//web//output//.akurasi_result.csv")
        data1 = pd.read_csv("D://Learning//python//web//output//Gaus_rpt.csv")
        st.write(data)     
        st.write(data1)

elif choose == "Contact":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')
