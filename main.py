import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

data1 = {
    'Provinsi': ['PAPUA', 'PAPUA BARAT', 'DKI JAKARTA', 'KEPULAUAN RIAU', 'KALIMANTAN TIMUR', 'MALUKU UTARA',
                 'KALIMANTAN BARAT', 'MALUKU', 'JAWA BARAT', 'KALIMANTAN TENGAH', 'BALI', 'SULAWESI UTARA',
                 'KALIMANTAN UTARA', 'NUSA TENGGARA BARAT', 'SUMATERA UTARA', 'KEPULAUAN BANGKA BELITUNG',
                 'DI YOGYAKARTA', 'KALIMANTAN SELATAN', 'JAWA TENGAH', 'ACEH', 'JAWA TIMUR', 'RIAU', 'SULAWESI TENGGARA',
                 'BANTEN', 'SUMATERA BARAT', 'JAMBI', 'BENGKULU', 'GORONTALO', 'SULAWESI SELATAN', 'NUSA TENGGARA TIMUR',
                 'SULAWESI TENGAH', 'SUMATERA SELATAN', 'LAMPUNG', 'SULAWESI BARAT'],
    'Value': [192.57, 124.82, 121.48, 115.97, 115.65, 110.60, 109.37, 107.97, 105.97, 104.77,
              104.74, 104.74, 104.69, 104.44, 103.40, 102.78, 102.37, 102.26, 100.63, 100.59,
              100.02, 99.21, 98.02, 97.72, 97.66, 96.84, 95.65, 95.28, 95.22, 93.69,
              92.50, 92.04, 90.46, 87.44],
    'Median Nilai Konstruksi yang Diselesaikan (rupiah)': [145000, 80000, 50000, 55000, 70000, 81400, 78000, 75000,
                                                           61000, 50000, 53100, 45000, 90000, 27000, 60000, 54000,
                                                           23000, 35000, 44000, 52000, 30000, 56000, 46000, 42700,
                                                           75000, 67250, 100000, 35000, 90000, 48000, 30000, 70000,
                                                           54250, 71000],
}

df = pd.DataFrame(data1)

features = ['Value', 'Median Nilai Konstruksi yang Diselesaikan (rupiah)']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

df['Category'] = df['Cluster'].map({0: 'Tinggi', 1: 'Rendah', 2: 'Sedang'})

st.title("Analisis IKK Provinsi")
st.title("Praktikum Geta My Love")
st.write("Masukkan nama provinsi untuk melihat detail dan perbandingannya.")

user_input = st.text_input("Masukkan nama provinsi:", "").strip().upper()

if user_input:
    if user_input in df['Provinsi'].values:
        selected_data = df[df['Provinsi'] == user_input]
        provinsi_value = selected_data['Value'].values[0]
        provinsi_category = selected_data['Category'].values[0]

        st.write(f"**{user_input}** memiliki nilai IKK sebesar **{provinsi_value}** dan termasuk kategori **{provinsi_category}**.")

        st.write("Berikut adalah perbandingan dengan provinsi lain:")
        plt.figure(figsize=(14, 7))
        sns.barplot(x='Provinsi', y='Value', data=df, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title("Perbandingan IKK Antar Provinsi", fontsize=16)
        plt.axhline(provinsi_value, color='red', linestyle='--', label=f'{user_input} ({provinsi_value})')
        plt.legend()
        st.pyplot(plt)

    else:
        st.error("Provinsi tidak ditemukan. Pastikan nama yang dimasukkan benar.")
