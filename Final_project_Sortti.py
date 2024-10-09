# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go

st.subheader('Final project for course Data Analytics and Mathematics')
st.markdown('Oulu University of Applied Sciences, Data Analytics and Project Management')
st.markdown('Mikko Sortti, Autumn 2024')
st.markdown('')
st.markdown('In this Final Project we import, clean, analyze and visualize data from home energy consumption, energy pool price and temperature.')
st.markdown('I have added my own consumption report and also my own actual flatrate prices for differend contract times. Aim is to analyze the data and see, if I have paid more or less, compared to pool prices.')
st.markdown('')
st.markdown('Also, aim is to analyze consumption habit with pool price, and see how consumption habit need to be changed in order to gain more savings.')
st.write('--------------------------------------------------------------------------------------------------------')

# Read the data to dataframes
df_consumption = pd.read_csv('consumption-report-own.csv', delimiter=';') #note, consumption data starts from 1.1.2020
df_elecprice = pd.read_csv('sahkon-hinta-010121-240924.csv', delimiter=',') #note, electricity price data starts from 1.1.2021
#df_consumption2 = pd.read_csv('Electricity_20-09-2024.csv', delimiter=';') #lets use our own consumption data from above, not this data


# First complete the following tasks
# - Change time format of both files to Pandas datetime
# - Join the two data frames according to time

# Format the Time -columns values to datatype Datetime in both of the dataframes
df_consumption['Time'] = pd.to_datetime(df_consumption['Time'], format = '%d.%m.%Y %H:%M')
df_elecprice['Time'] = pd.to_datetime(df_elecprice['Time'], format = '%d-%m-%Y %H:%M:%S')

# Merge the two dataframes according to time
df_merged = pd.merge(df_consumption, df_elecprice, on='Time', how='outer')

# Rename the columns to be easier to use and refer in the code
df_merged.rename(columns={'Price (cent/kWh)': 'Price', 'Energy (kWh)': 'Energy'}, inplace=True)
#df_merged.head(40)

# I noted from the data, that the price data starts from 2021 and consumption data from 2020.
# I tried to use porssisahko.net API to fetch 2020 price data, and added conditions that it would only check and fetch data, if price would be NaN.
# Code itself seemed to work, but after first run, when there were still NaN values, I checked that the porssisahko.net data starts also only from 2021..

# START of time wasted, but the code is good
# lets update the price, where price information isn't available

# Function to fetch the price from Porssisahko.net via API
# def fetch_price(date, hour):
#     url =f"https://api.porssisahko.net/v1/price.json?date={date}&hour={hour}"
#     response = requests.get(url) #use API and return response
#     print('Response is ', response)
#     if response.status_code == 200: #If return is successful
#         data = response.json() #Extract the json response
#         print('and Data is: ', data.get('price'))
#         return data.get('price')  # Returns the price from the API response
#     return None #if unsuccessful, return nothing
    
# Function to update the price column only for NaN values
# def update_prices(df, num_rows=None):
      # First run took 53 minutes. Had to apply option to run for few rows for testing purposes
#     if num_rows is not None:
#         df_subset = df.iloc[:num_rows]  # Select the first 'num_rows' rows
#     else:
#         df_subset = df  # Process the entire DataFrame if num_rows is not specified

#     for index, row in df_subset.iterrows():
#         if pd.isna(row['Price']):  # Check if the price is NaN
#             date = row['Time'].strftime('%Y-%m-%d')  # Format the date for API
#             hour = row['Time'].hour  # Get the hour for API
#             print('Date is ', date, ' and hour is ', hour)
#             price = fetch_price(date, hour)  # Fetch the price via fetch_price function
#             print('Price is ', price)
#             df.at[index, 'Price'] = price  # Update the DataFrame

# Update prices in the DataFrame
#update_prices(df_merged, 50)

# Print the updated DataFrame
#print(df_merged)
# END of the time wasted, but code was good

# Insted, let's clean the data rows from NaN values, that the data would be coherent.
# By doing this, it should be taken into consideration that 49 (I checked the dataframe with DataWrangler plugin) rows will be deleted after the time 1.1.2021, so some hour data/rows won't be present. The NaN values were from 2024-09-23 00:00 to 2024-09-24 23:00, so 48 hours. 
# Other option might have been to accumulate data syntetically, but they would not have been real data and could have skewed the analytics result.

# Also need to check the datatypes and handle possible "," and "." types in floats/numbers.

# lets clean the data(frame), and drop every row with NaN data
df_cleaned = df_merged.dropna()
df_cleaned = df_cleaned.copy()

# Also, let's check and clean the datatypes
#print(df_cleaned.dtypes)
# Energy and Tmperature are object and not numeric.
# We probably need to address and sort the , and . problem in floats in python
# Lets replace , with . for Python to understand
df_cleaned['Energy'] = df_cleaned['Energy'].str.replace(',', '.')
df_cleaned['Temperature'] = df_cleaned['Temperature'].str.replace(',', '.')

# Now convert them to numeric types
df_cleaned['Energy'] = pd.to_numeric(df_cleaned['Energy'], errors='coerce')
df_cleaned['Temperature'] = pd.to_numeric(df_cleaned['Temperature'], errors='coerce')

# Verify the changes - it worked
#print(df_cleaned[['Energy', 'Temperature']].head())

#print(df_cleaned.dtypes)

# Im interested to know, if I have saved money or not, by being in flat rate electricity contract all these years.
# I have saved the information of my contract prices, and will add them as new column. Price will be determined by Time column and compared.
# Datetime can be compared, so no need to do epoch time, but that would have been possibility/option also. 

# Adding my own actual electricity price as a new column.
# My electricity flat rate prices / contract with dates
# From 1.1.2021 to 11.2.2022 = 4,99c/kWh
# From 12.2.2022 to 30.11.2022 = 7,75c/kWh
# From 1.12.2022 to 30.4.2023 = 6,88c/kWh
# From 1.5.2023 to 11.2.2024 = 7,75c/kWh
# From 12.2.2024 to present = 7,99c/kWh

# Function to assign electricity price based on date
def get_flatrate(time):
    if time < pd.Timestamp('2022-02-12'):
        return 4.99
    elif pd.Timestamp('2022-02-12') <= time < pd.Timestamp('2022-12-01'):
        return 7.75
    elif pd.Timestamp('2022-12-01') <= time < pd.Timestamp('2023-05-01'):
        return 6.88
    elif pd.Timestamp('2023-05-01') <= time < pd.Timestamp('2024-02-12'):
        return 7.75
    elif time >= pd.Timestamp('2024-02-12'):
        return 7.99
    else:
        return np.nan  # In case of an unexpected value

# Add flatrate column and value based on the timeframe / contract price
df_cleaned = df_cleaned.copy() # Let's make deep copy of the dataframe, so Python wont warn about "SettingWithCopyWarning"
df_cleaned['Flatrate'] = df_cleaned['Time'].apply(get_flatrate)

# Display the updated DataFrame
#print(df_cleaned)

# - Calculate the hourly bill paid (using information about the price and the consumption) - compared to pool price and flatrate
# Calculate the total cost with the flat rate
df_cleaned['FlatRateCost'] = df_cleaned['Energy'] * df_cleaned['Flatrate']

# Calculate the total cost with the pool price
df_cleaned['PoolPriceCost'] = df_cleaned['Energy'] * df_cleaned['Price']

# Did it work? Yes
#print(df_cleaned[['Time', 'Energy', 'Price', 'Flatrate', 'FlatRateCost', 'PoolPriceCost']])

# Summing up the total cost for each
total_flat_rate_cost = df_cleaned['FlatRateCost'].sum() / 100
total_pool_price_cost = df_cleaned['PoolPriceCost'].sum() / 100

# - Calculated grouped values of daily, weekly or hourly consumption, bill, average price and average temperature
# - I think that to me, most interesting grouping for data would be by month. Let's do that.

# Grouped by month (1- 12)
# Extract month from the 'Time' column
df_cleaned['Month'] = df_cleaned['Time'].dt.month

# Group by the extracted month, sum up the month consumption to count average values of the months
monthly_sums = df_cleaned.groupby('Month').agg({
    'Energy': 'sum',  # Sum of energy consumption for each month
    'Flatrate': 'mean',  # Average flatrate price for each month
    'Price': 'mean',       # Average pool price for each month
    'Temperature': 'mean'  # Average temperature for each month
}).rename(columns={
    'Energy': 'TotalEnergyConsumption',
    'Flatrate': 'AvgFlatrate',
    'Price': 'AvgPoolPrice',
    'Temperature': 'AvgTemperature'
})

# Calculate the average monthly energy consumption from the total sums
avg_consumption_by_month = monthly_sums['TotalEnergyConsumption'] / len(df_cleaned['Time'].dt.year.unique())

# Add the average consumption by month to the dataframe
monthly_sums['AvgMonthlyConsumption'] = avg_consumption_by_month

# Calculate the average cost of the monthly consumption with pool price and flat rate
monthly_sums['CostWithPoolPrice'] = monthly_sums['AvgMonthlyConsumption'] * monthly_sums['AvgPoolPrice'] / 100
monthly_sums['CostWithFlatRate'] = monthly_sums['AvgMonthlyConsumption'] * monthly_sums['AvgFlatrate'] / 100


# Display the grouped data (monthly) and the average monthly consumption
#print(monthly_sums[['AvgMonthlyConsumption', 'CostWithPoolPrice', 'CostWithFlatRate']])

# - Calculated grouped values of daily, weekly or hourly consumption, bill, average price and average temperature
# - Next I am interested of average consumption of electricity, average pool price and average flat rate price, by the hour 0 - 23

# Extract the hour (0-23) from the 'Time' column
df_cleaned['Hour'] = df_cleaned['Time'].dt.hour

# Group by the hour (0-23) and calculate the average for electricity consumption, pool price, and flat rate
hourly_avg = df_cleaned.groupby('Hour').agg({
    'Energy': 'mean',  # Average electricity consumption per hour
    'Flatrate': 'mean',     # Average flat rate per hour
    'Price': 'mean'         # Average pool price per hour
}).rename(columns={
    'Energy': 'AvgHourlyConsumption',
    'Flatrate': 'AvgHourlyFlatrate',
    'Price': 'AvgHourlyPoolPrice'
})

# Display the grouped data
#print(hourly_avg)

# Create a visualization which includes
# - A selector for time interval included in the analysis
# - Consumption, bill, average price and average temperature over selected period
# - Selector for grouping interval 
# - Line graph of consumption, bill, average price and average temperature over the range selected using the grouping interval selected.
# - Bonus: visualization that shows when have I saved and when over paid (pool price vs. flatrate)

# STREAMLIT part

# Sidebar with navigation options
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose a section", ["Grouped data", "Price analysis", "Usage analysis"])

# Function to group the data
def group_data(df, group_by):
    if group_by == 'Day':
        grouped = df.resample('D', on='Time').agg({'Energy': 'sum', 'Temperature': 'mean'}) # Group by day, sum energy, mean temperature
    elif group_by == 'Week':
        grouped = df.resample('W', on='Time').agg({'Energy': 'sum', 'Temperature': 'mean'}) # Group by Week, sum energy, mean temperature
    elif group_by == 'Month':
        grouped = df.resample('M', on='Time').agg({'Energy': 'sum', 'Temperature': 'mean'}) # Group by Month, sum energy, mean temperature
    return grouped

# Navigation: what happens with each selected option
if options == "Grouped data":
    # Selection for grouping
    group_by_option = st.selectbox("Group data by", ["Day", "Week", "Month"])

    # Group and aggregate the data with function
    grouped_data = group_data(df_cleaned, group_by_option)

    # Plot the grouped data with temperature on a secondary y-axis
    st.subheader(f'{group_by_option} Energy Consumption and Temperature')

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(grouped_data.index, grouped_data['Energy'], label='Energy Consumption (kWh, sum)', color='b')
    ax1.set_xlabel(f'{group_by_option}')
    ax1.set_ylabel('Energy Consumption (kWh)', color='b')
    
    ax2 = ax1.twinx()  # Create a second y-axis for average temperature
    ax2.plot(grouped_data.index, grouped_data['Temperature'], label='Temperature (C, average)', color='r')
    ax2.set_ylabel('Temperature (°C)', color='r')

    # Display the plot
    plt.title(f'{group_by_option} Energy Consumption and Temperature')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    st.pyplot(fig)

    st.write("Analysis: There seem's to be clear correlation between temperature and consumption of energy.")

elif options == "Price analysis":
    st.subheader("Price Analysis")
    st.write("Price analysis: pool price vs. flatrate by hourly average, with hourly average consumption data.")
       
    # Create the figure and axes for plotting
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bar plot for AvgHourlyConsumption
    ax1.bar(hourly_avg.index, hourly_avg['AvgHourlyConsumption'], color='b', label='AvgHourlyConsumption (kWh)')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('AvgHourlyConsumption (kWh)', color='b')

    # Create a second y-axis for both AvgHourlyFlatrate and AvgHourlyPoolPrice
    ax2 = ax1.twinx()

    # Line plot for AvgHourlyFlatrate on second y-axis
    ax2.plot(hourly_avg.index, hourly_avg['AvgHourlyFlatrate'], color='g', label='AvgHourlyFlatrate (c/kWh)')
    # Line plot for AvgHourlyPoolPrice on second y-axis
    ax2.plot(hourly_avg.index, hourly_avg['AvgHourlyPoolPrice'], color='r', label='AvgHourlyPoolPrice (c/kWh)')
    # Fill the area where AvgHourlyPoolPrice is greater than AvgHourlyFlatrate
    ax2.fill_between(hourly_avg.index, hourly_avg['AvgHourlyFlatrate'], hourly_avg['AvgHourlyPoolPrice'],
                 where=(hourly_avg['AvgHourlyPoolPrice'] > hourly_avg['AvgHourlyFlatrate']),
                 color='green', alpha=0.3, label='Pool Price > Flat Rate')


    # Set the second y-axis labels and colors
    ax2.set_ylabel('Price (c/kWh)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine the legends of both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot
    st.pyplot(fig)

    # Monthly data calculated before
    st.subheader("Monthly Average Consumption and Price Comparison")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for average sum of consumption by month
    ax1.bar(monthly_sums.index, monthly_sums['AvgMonthlyConsumption'], color='b', alpha=0.6, label='Avg Monthly Consumption (kWh)')

    # Line plot for average pool price and flatrate price
    ax2 = ax1.twinx()  # Second y-axis sharing the same axis

    ax2.plot(monthly_sums.index, monthly_sums['AvgPoolPrice'], color='r', label='Avg Pool Price (c/kWh)')
    ax2.plot(monthly_sums.index, monthly_sums['AvgFlatrate'], color='g', label='Avg Flat Rate (c/kWh)')

    # Set x-axis labels and title
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Avg Monthly Consumption (kWh)', color='b')
    ax2.set_ylabel('Price (c/kWh)', color='r')

    # Add legends for both plots
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Cost calculations
    flatrate_price = total_flat_rate_cost  # Total price paid with flatrate
    pool_price = total_pool_price_cost  # Calculated total cost based on pool price

    # Calculate the difference
    difference = flatrate_price - pool_price
    if difference > 0:
        result_text = f"I have paid {difference: .2f} more with the flat rate."
    else:
        result_text = f"I have saved {abs(difference): .2f} euros by using the flat rate."

    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Total", 
        orientation="v",
        measure=["absolute", "absolute", "relative"],  # Use absolute for flat and pool prices, relative for difference
        x=["Flat Rate Cost", "Pool Price Cost", "Difference"],
        text=[f"{flatrate_price: .2f} €", f"{pool_price: .2f} €", f"{difference: .2f} €"],
        y=[flatrate_price, pool_price, difference],  # Show the actual pool price and difference
        connector={"line":{"color":"rgb(63, 63, 63)"}},
        decreasing={"marker":{"color":"green"}},
        increasing={"marker":{"color":"red"}},
        totals={"marker":{"color":"blue"}}
    ))

    # Customize the layout
    fig.update_layout(
        title="Cost Comparison: Flatrate vs Pool Price",
        waterfallgap=0.3,
        yaxis=dict(title='Price (€)')
    )

    # Display the waterfall chart in Streamlit
    st.subheader("Electricity cost Comparison")
    st.plotly_chart(fig)

    # Write the result
    st.write(result_text)
    # Write the result and analysis/conclusion
    st.write(f"Total paid from energy with flat rate: {total_flat_rate_cost:.2f} euros")
    st.write(f"Total would have paid from energy with pool price: {total_pool_price_cost:.2f} euros")
    st.write('Analysis: Flatrate contracts have been more cheaper with this consumption habit, so far.')


elif options == "Usage analysis":
    st.subheader("Usage Analysis")
    st.write("Usage analysis with average pool price: analysis")
    
    # Create the figure and axes for plotting
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bar plot for AvgHourlyConsumption
    ax1.bar(hourly_avg.index, hourly_avg['AvgHourlyConsumption'], color='b', label='AvgHourlyConsumption (kWh)')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('AvgHourlyConsumption (kWh)', color='b')

    # Create a second y-axis for both AvgHourlyFlatrate and AvgHourlyPoolPrice
    ax2 = ax1.twinx()

    threshold = 7.99 #current flatrate price
    # Line plot for AvgHourlyPoolPrice on second y-axis with fill
    ax2.plot(hourly_avg.index, hourly_avg['AvgHourlyPoolPrice'], color='r', label='AvgHourlyPoolPrice (c/kWh)')
    ax2.fill_between(hourly_avg.index, hourly_avg['AvgHourlyPoolPrice'], threshold, 
                 where=(hourly_avg['AvgHourlyPoolPrice'] < threshold), color='red', alpha=0.3) #fill area that has lower pool price than my current flatrate

    # Set the second y-axis labels and colors
    ax2.set_ylabel('Price (c/kWh)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine the legends of both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot
    st.pyplot(fig)

    # Write the analysis
    st.write(f'Analysis: My average flatrate price so far has been {df_cleaned["Flatrate"].mean(): .2f} c/kWh, but current flatrate is 7,99 c/kWh.')
    st.write(f'From the data sample, average pool price has been {df_cleaned["Price"].mean(): .2f} c/kWh.')
    st.write('I would need to change my energy consumption habit towards more hours from 23 to 05 (filled area), when price would be below current, and lower my consumption from other times as much as possible.')
    st.write('As for the moment, it seems nearly impossible to do drastic change to the consumption habit. Conclusion: try to continue using flatrate contracts, which price is below the average pool price.')