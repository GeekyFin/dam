import pandas as pd
import streamlit as st
import plotly.express as px
import os

#29. Use the same population data as in the previous exercise. 

# Get the current working directory (directory of the script)
dir_path = os.path.dirname(os.path.realpath(__file__))

# Construct the full path to the CSV file
csv_path = os.path.join(dir_path, 'population.csv')

population_df = pd.read_csv(csv_path)

#a) Create a new Pandas data frame where the first column is the year and other 142 columns are populations of all the countries in the data
# year countrya countryb countryc
# 1952 123123   1231231  131231231
#pivot the original dataframe
#pivoted_df = population_df.pivot(index='year', columns='country', values='pop')
pivoted_df = population_df.pivot(index='year', columns='country', values='pop').reset_index()


#b) Use streamlit and create an interactive web graph where you can select the countries to be included in the population plot
# Streamlit app
st.title('Interactive Population Graph')

# Select countries to display
countries = st.multiselect('Select countries to plot', options=pivoted_df.columns[1:], default=pivoted_df.columns[1:])

# Filter DataFrame to include selected countries
df_filtered = pivoted_df[['year'] + countries]

# Plot using Plotly Express
fig = px.line(df_filtered, x='year', y=countries, title='Population Over Time')

# Show plot
st.plotly_chart(fig)