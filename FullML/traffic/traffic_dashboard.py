import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load data
df = pd.read_csv('traffic.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.drop('ID', axis=1)

# Sidebar for user input
junctions = df['Junction'].unique()
selected_junction = st.sidebar.selectbox('Select Junction', junctions)

# Filter data based on the selected junction
df_junction = df[df['Junction'] == selected_junction]

# Plot the traffic data
st.title(f'Traffic Data for Junction {selected_junction}')
st.line_chart(df_junction.set_index('DateTime')['Vehicles'])

# Feature engineering
df_junction['Year'] = df_junction['DateTime'].dt.year
df_junction['Month'] = df_junction['DateTime'].dt.month
df_junction['Date_no'] = df_junction['DateTime'].dt.day
df_junction['Hour'] = df_junction['DateTime'].dt.hour
df_junction['Day'] = df_junction['DateTime'].dt.strftime("%A")

# Show the correlation heatmap
if st.sidebar.checkbox('Show Correlation Heatmap'):
    df_numerical = df_junction.select_dtypes(include=['number'])
    corrmat = df_numerical.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corrmat, cmap="coolwarm", annot=True, square=True)
    st.pyplot(plt.gcf())

# Show ACF and PACF plots
if st.sidebar.checkbox('Show ACF and PACF plots'):
    st.write('ACF and PACF plots')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(df_junction['Vehicles'].values, ax=axes[0])
    plot_pacf(df_junction['Vehicles'].values, ax=axes[1])
    st.pyplot(fig)

# ARIMA Model
if st.sidebar.checkbox('Run ARIMA model'):
    p = st.sidebar.number_input('Enter p value', min_value=0, value=24)
    d = st.sidebar.number_input('Enter d value', min_value=0, value=1)
    q = st.sidebar.number_input('Enter q value', min_value=0, value=8)
    model = ARIMA(df_junction['Vehicles'].values, order=(p, d, q))
    model_fit = model.fit()
    st.write(model_fit.summary())
    st.line_chart(model_fit.predict(5))

# Show pairplot
if st.sidebar.checkbox('Show Pairplot'):
    sns.pairplot(data=df_junction, hue='Junction')
    st.pyplot(plt.gcf())

# Show countplot
if st.sidebar.checkbox('Show Countplot'):
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df_junction, x='Year', hue='Junction')
    st.pyplot(plt.gcf())

st.sidebar.title('Settings')
st.sidebar.markdown('Select options to view different plots and analysis.')
