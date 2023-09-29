# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:16:35 2023

@author: ragha
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_bank = pd.read_csv("bank.csv")

with st.sidebar:
    selected = option_menu("Menu", ["Home", "Display dataframe" ,"Pairplot"],
                          default_index=0,
                          orientation="vertical",
                          styles={"nav-link": {"font-size": "22px", "text-align": "left", "margin": "2px", 
                                                "--hover-color": "#0000FF"},
                                   "container" : {"max-width": "4000px"},
                                   "nav-link-selected": {"background-color": "#0000FF"}})

if selected == "Home":
        st.write("# This app displays information about the bank marketing dataset")
        
elif selected == "Display dataframe":
        st.write("Displaying my dataframe below")
        st.write(df_bank)

elif selected == "Pairplot":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        sns.pairplot(df_bank, hue='deposit')
        st.pyplot()