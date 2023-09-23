# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:45:10 2023

@author: raghavi
"""
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df.drop("Unnamed: 32", axis=1, inplace=True)
numeric_cols = list(df.select_dtypes(include=['int', 'float']).columns)
df['diagnosis'] = df[['diagnosis']].applymap(lambda x: 0 if x=='B' else 1)
df['diagnosis'] = pd.to_numeric(df['diagnosis'])

corr = df.corr()
a = corr.diagnosis[(corr['diagnosis'] > 0.7) | (corr['diagnosis'] < -0.7)]
cols_max_corr=list(a.index)
df_with_max_corr = df[cols_max_corr]

with st.sidebar:
    selected = option_menu("Menu", ["Home", "Observation from the dataset" ,"Change the value", "Other Plots"],
                          default_index=0,
                          orientation="vertical",
                          styles={"nav-link": {"font-size": "22px", "text-align": "left", "margin": "2px", 
                                                "--hover-color": "#0000FF"},
                                   "container" : {"max-width": "4000px"},
                                   "nav-link-selected": {"background-color": "#0000FF"}})

if selected == "Home":
        st.write("# This app displays EDA for the Breast cancer Dataset and in the future we can explore further to perform prediction as well")
        
elif selected == "Observation from the dataset":
        st.write("There are no null values in the dataset and there are no duplicates")
        st.write("Once I read the dataset, there was a column by the name Unnamed: 32 which is not a good column name. So we are dropping that column")
        st.write("There is one categorical variable which is 'diagnosis' and the other columns are all continuous variables. Also, if we look further into the min and max values for the columns, we can say that the columns are of different scales.")


elif selected == "Change the value":
        st.set_option('deprecation.showPyplotGlobalUse', False)

        x_axis = st.selectbox(
    'choose a column for the x-axis to plot',
    ('radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst')
)
        y_axis = st.selectbox(
    'choose a column for the y-axis to plot',
    ('radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst')
)
        fig1 = sns.lmplot(data=df, x=x_axis, y=y_axis, hue='diagnosis')
        st.pyplot(fig1)
        
elif selected == "Other Plots":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        fig, ax = plt.subplots(2,2, figsize=(10,6))
        plt.subplots_adjust(wspace=0.5)
        
        sns.boxplot(x= "variable" ,y = "value", hue= "diagnosis", 
                    data=pd.melt(df[['radius_mean','radius_worst','diagnosis']], id_vars='diagnosis'),ax = ax[0,0])
        
        sns.boxplot(x= "variable" ,y = "value", hue= "diagnosis", 
                    data=pd.melt(df[['perimeter_mean','perimeter_worst','diagnosis']], id_vars='diagnosis'),ax = ax[1,0])

        sns.barplot(x= "variable" ,y = "value", hue= "diagnosis",
                    data=pd.melt(df[['radius_mean','radius_worst','diagnosis']], id_vars='diagnosis'),ax = ax[0,1])
        
        sns.barplot(x= "variable" ,y = "value", hue= "diagnosis",
                    data=pd.melt(df[['perimeter_mean','perimeter_worst','diagnosis']], id_vars='diagnosis'),ax = ax[1,1])

        st.pyplot()
        st.write("In the above boxplots, we see that the radius_worst and perimeter_worst are more linearly separable when compared to the radius_mean and perimeter_mean")
        
        fig, ax = plt.subplots(2,2, figsize=(10,6))
        plt.subplots_adjust(wspace=0.5)
        
        sns.boxplot(x= "variable" ,y = "value", hue= "diagnosis", 
                    data=pd.melt(df[['area_mean','area_worst','diagnosis']], id_vars='diagnosis'),ax = ax[0,0])
        
        sns.boxplot(x= "variable" ,y = "value", hue= "diagnosis", 
                    data=pd.melt(df[['concave points_mean','concave points_worst','diagnosis']], id_vars='diagnosis'),ax = ax[1,0])
        
        sns.stripplot(x= "variable" ,y = "value", hue= "diagnosis", jitter=True, palette='Set1',
                    data=pd.melt(df[['area_mean','area_worst','diagnosis']], id_vars='diagnosis'),ax = ax[0,1])
        
        sns.stripplot(x= "variable" ,y = "value", hue= "diagnosis", jitter=True, palette='Set1',
                    data=pd.melt(df[['concave points_mean','concave points_worst','diagnosis']], id_vars='diagnosis'),ax = ax[1,1])
        
        st.pyplot()
        
        st.write("In the above boxplots, we see that the area_worst and concave points_worst are more linearly separable when compared to the area_mean and concave points_mean")
        st.write("From the above box plots, we can conclude that concave points_worst, area_worst, perimeter_worst and radius_worst are even better predictors of cancer than the other columns")        
