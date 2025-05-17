import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,  RobustScaler
from streamlit_option_menu import option_menu
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

# CSS untuk mengubah warna sidebar menjadi biru muda
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #D6E6F2; /* Warna Biru Muda */
        }
        [data-testid="stSidebarNav"] span {
            color: black; /* Warna teks */
            font-size: 18px;
        }
        [data-testid="stSidebarNav"] {
            padding-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Menu
with st.sidebar:
    
    selected = option_menu(
        "Clustering UMKM Kabupaten Bojonegoro",
        ["Beranda", "Preprocessing", "Pemodelan K-Means", "Hasil Clustering", "Evaluasi",'Analisa Hasil'], 
        icons=["house", "gear", "bar-chart", "list", "check-circle","clipboard-check"],
        #menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "black", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "2px"},
            "nav-link-selected": {"background-color": "#4A90E2", "color": "white", "font-weight": "bold"},
        }
    )

# Halaman Beranda dengan Upload File
if selected == "Beranda":
    st.title("Beranda")
    st.write("Selamat datang di aplikasi Clustering UMKM Kabupaten Bojonegoro!")

    # Header Upload Data
    st.header("Upload Data UMKM")

    # Jika sudah ada data yang tersimpan di session_state, tampilkan preview
    if "uploaded_data" in st.session_state:
        st.success("\u2705 Data sudah tersedia di sistem!")
        st.write("### Preview Data:")
        st.dataframe(st.session_state["uploaded_data"])

        # Opsi untuk mengunggah ulang data
        if st.checkbox("Unggah data baru"):
            uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"], key="new_upload")

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file, engine="openpyxl")

                    st.success("\u2705 File baru berhasil diunggah!")
                    st.write("### Preview Data Baru:")
                    st.dataframe(df.head())

                    # Perbarui session_state dengan data baru
                    st.session_state["uploaded_data"] = df

                except Exception as e:
                    st.error(f"\u26A0 Terjadi kesalahan saat membaca file: {e}")

    else:
        # Jika belum ada data, langsung minta upload file
        uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, engine="openpyxl")

                st.success("\u2705 File berhasil diunggah!")
                st.write("### Preview Data:")
                st.dataframe(df.head())

                # Simpan data ke dalam session_state agar tidak perlu upload ulang
                st.session_state["uploaded_data"] = df

            except Exception as e:
                st.error(f"\u26A0 Terjadi kesalahan saat membaca file: {e}")

# Halaman Preprocessing dengan Transformasi Label Encoding
elif selected == "Preprocessing":
    st.title("Preprocessing")

    if "uploaded_data" in st.session_state:
        df = st.session_state["uploaded_data"].copy()
        st.write("### Data yang akan diproses:")
        st.dataframe(df.head())

        # Pilihan Tab untuk Transformasi dan Normalisasi
        tab1, tab2 = st.tabs(["Transformasi Data", "Normalisasi Data"])

        # ==============================
        # TRANSFORMASI DATA
        # ==============================
        
        with tab1:
            st.subheader("Transformasi Data")

            required_columns = ["Sektor Usaha", "Posisi Pasar"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.warning(f"\u26A0 Kolom berikut tidak ditemukan dalam data: {', '.join(missing_columns)}")
            else:
                # Hapus baris yang memiliki NaN pada "Sektor Usaha" dan "Posisi Pasar"
                df_cleaned = df.dropna(subset=["Sektor Usaha", "Posisi Pasar"]).copy()

                if st.button("Lakukan Transformasi"):
                    df_transformed = df.copy()

                    # Transformasi "Sektor Usaha"
                    sektor_mapping = {"Industri Pengolahan": 1, "Perdagangan": 2, "Jasa": 3}
                    df_transformed["Sektor Usaha"] = df_transformed["Sektor Usaha"].map(sektor_mapping)

                    # Transformasi "Posisi Pasar"
                    posisi_mapping = {"Cukup": 1, "Baik": 2, "Sangat Baik": 3}
                    df_transformed["Posisi Pasar"] = df_transformed["Posisi Pasar"].map(posisi_mapping)

                    st.success("\u2705 Transformasi selesai!")
                    st.write("### Data Setelah Transformasi")
                    st.dataframe(df_transformed.head())

                    mapping_df = pd.DataFrame([
                        {"Kolom": "Sektor Usaha", "Kategori": k, "Encoded Value": v} for k, v in sektor_mapping.items()
                    ] + [
                        {"Kolom": "Posisi Pasar", "Kategori": k, "Encoded Value": v} for k, v in posisi_mapping.items()
                    ])

                    st.write("### Mapping Encoding")
                    st.dataframe(mapping_df)

                    st.session_state["encoded_data"] = df_transformed

        # ==============================
        # NORMALISASI DATA
        # ==============================
        with tab2:
            st.subheader("Normalisasi Data")
            st.write("Normalisasi menggunakan Min-Max Scaling akan diterapkan pada seluruh kolom numerik.")

            if "encoded_data" in st.session_state:
                df_transformed = st.session_state["encoded_data"].copy()
            else:
                df_transformed = df.copy()

            # Pilih hanya kolom numerik
            numeric_columns = df_transformed.select_dtypes(include=["int64", "float64"]).columns.tolist()

            if numeric_columns:
                if st.button("Lakukan Normalisasi"):
                    df_norm = df_transformed.copy()

                    # Normalisasi menggunakan Min-Max Scaling untuk semua kolom numerik
                    scaler = MinMaxScaler()
                    df_norm[numeric_columns] = scaler.fit_transform(df_norm[numeric_columns])

                    df_norm[numeric_columns] = df_norm[numeric_columns].round(4)
                    
                    st.success("\u2705 Normalisasi selesai!")
                    st.write("### Data Setelah Normalisasi")
                    st.dataframe(df_norm.head())

                    # Simpan hasil normalisasi ke session state
                    st.session_state["normalized_data"] = df_norm

            else:
                st.warning("\u26A0 Tidak ada kolom numerik untuk dinormalisasi.")
    else:
        st.warning("\u26A0 Silakan unggah data terlebih dahulu di Beranda.")

# Halaman Pemodelan K-Means
elif selected == "Pemodelan K-Means":
    st.title("Pemodelan K-Means")
    st.write("Proses pemodelan clustering menggunakan algoritma K-Means.")

    if "normalized_data" in st.session_state:
        df = st.session_state["normalized_data"].copy()
        st.write("### Data yang akan digunakan untuk Clustering:")
        st.dataframe(df.head())

        # Pilih hanya kolom numerik untuk clustering
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        if not numeric_columns:
            st.warning("⚠️ Tidak ada kolom numerik yang tersedia untuk clustering.")
        else:
            # Hapus baris dengan nilai NaN sebelum pemodelan
            df_cleaned = df[numeric_columns].dropna()

            # ===============================
            # Menentukan Jumlah Cluster (Elbow Method)
            # ===============================
            st.subheader("Menentukan Jumlah Cluster (Elbow Method)")
            
            inertia_values = []
            cluster_range = range(1, 11)  # Uji klaster dari 1 hingga 10
            
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
                kmeans.fit(df_cleaned)
                inertia_values.append(kmeans.inertia_)

            # Visualisasi Elbow Method
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(cluster_range, inertia_values, marker="o", linestyle="-")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Method untuk Menentukan Jumlah Cluster")
            st.pyplot(fig)


            # Pilih jumlah cluster
            n_clusters = st.slider("Pilih jumlah cluster:", min_value=2, max_value=10,)

            # ===============================
            # Pemodelan K-Means
            # ===============================
            st.subheader(f"Pemodelan K-Means dengan {n_clusters} Cluster")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, init='k-means++')
            df_cleaned["Cluster"] = kmeans.fit_predict(df_cleaned) + 1 #Mulai cluster dari angka 1

            st.success("\u2705 Clustering selesai!")
            st.write("### Hasil Clustering")
            st.dataframe(df_cleaned.head())

            # ===============================
            # Simpan hasil clustering
            # ===============================
            st.session_state["clustered_data"] = df_cleaned
            st.session_state["kmeans_model"] = kmeans

    else:
        st.warning("\u26A0 Silakan lakukan preprocessing terlebih dahulu di menu Preprocessing.")

# Halaman Hasil Clustering
elif selected == "Hasil Clustering":
    st.title("Hasil Clustering")
    st.write("Menampilkan hasil clustering dari data UMKM.")

    if "clustered_data" in st.session_state and "uploaded_data" in st.session_state:
        df_clustered = st.session_state["clustered_data"].copy()
        df_original = st.session_state["uploaded_data"].copy()

        # Gabungkan hanya kolom nama usaha dan hasil clustering
        df_result = pd.concat([
            df_original[["Nama Usaha"]].reset_index(drop=True), 
            df_clustered[["Cluster"]].reset_index(drop=True)
        ], axis=1)

        st.write("### Hasil Clustering")
        st.dataframe(df_result)

        # Hitung jumlah masing-masing cluster
        cluster_counts = df_result["Cluster"].value_counts().sort_index().reset_index()
        cluster_counts.columns = ["Cluster", "Jumlah UMKM"]

        st.write("### Jumlah UMKM pada Setiap Cluster")
        st.dataframe(cluster_counts)

        # Visualisasi jumlah cluster
        
        if not cluster_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(cluster_counts["Cluster"], cluster_counts["Jumlah UMKM"], color="#4A90E2")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Jumlah UMKM")
            ax.set_title("Jumlah UMKM pada Setiap Cluster")
            st.pyplot(fig)
    else:
        st.warning("\u26A0 Belum ada hasil clustering yang tersedia. Silakan lakukan pemodelan K-Means terlebih dahulu.")

# Halaman Evaluasi
import warnings

if selected == "Evaluasi":
    st.title("Evaluasi Cluster")

    if "clustered_data" in st.session_state:
        df_clustered = st.session_state["clustered_data"].copy()

        if "Cluster" in df_clustered.columns and not df_clustered.empty:
            X = df_clustered.drop(columns=["Cluster"])
            labels = df_clustered["Cluster"]

            # Tab hanya untuk SC dan DBI
            tab1, tab2 = st.tabs(["Silhouette Coefficient", "Davies-Bouldin Index"])

            with tab1:
                st.subheader("Evaluasi dengan Silhouette Coefficient")

                # Tentukan jumlah cluster saat ini dari hasil labels
                k_terpilih = len(set(labels))

                if k_terpilih > 1:
                    silhouette_scores = []
                    k_values = range(2, 11)

                    warnings.simplefilter("ignore")

                    for k in k_values:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
                        cluster_labels = kmeans.fit_predict(X)
                        silhouette_avg = silhouette_score(X, cluster_labels)
                        silhouette_scores.append((k, silhouette_avg))

                    # Simpan ke DataFrame
                    df_silhouette = pd.DataFrame(silhouette_scores, columns=["Jumlah Cluster (k)", "Silhouette Coefficient"])

                    # Ambil nilai silhouette yang sesuai dengan jumlah cluster terpilih
                    nilai_silhouette_terpilih = df_silhouette[df_silhouette["Jumlah Cluster (k)"] == k_terpilih]["Silhouette Coefficient"].values[0]

                    # Tampilkan hasil
                    st.success(f"\u2705 Silhouette Coefficient untuk {k_terpilih} cluster: {nilai_silhouette_terpilih:.4f}")

                    # Visualisasi
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df_silhouette["Jumlah Cluster (k)"], df_silhouette["Silhouette Coefficient"], 'bo-', marker='o')
                    ax.set_xlabel("Jumlah Cluster (k)")
                    ax.set_ylabel("Silhouette Coefficient")
                    ax.set_title("Silhouette Coefficient untuk berbagai jumlah cluster (2-10)")
                    ax.grid(True)
                    st.pyplot(fig)

                    # Tampilkan tabel
                    st.write("### Hasil Silhouette Coefficient untuk Cluster 2 sampai 10")
                    st.write(df_silhouette)

                else:
                    st.warning("Jumlah cluster terlalu sedikit untuk menghitung Silhouette Score.")


            with tab2:
                st.subheader("Evaluasi dengan Davies-Bouldin Index (DBI)")

                # Skala data
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)

                # Tentukan jumlah cluster yang digunakan saat ini dari hasil label
                k_terpilih = len(set(labels))

                if k_terpilih > 1:
                    # Hitung semua nilai DBI untuk berbagai jumlah cluster
                    dbi_scores = []
                    for k in k_values:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
                        cluster_labels = kmeans.fit_predict(X_scaled)
                        dbi_score = davies_bouldin_score(X_scaled, cluster_labels)
                        dbi_scores.append((k, dbi_score))

                    # Simpan hasil ke DataFrame
                    df_dbi = pd.DataFrame(dbi_scores, columns=["Jumlah Cluster (k)", "Davies-Bouldin Index"])

                    # Cari nilai DBI yang sesuai dengan k terpilih
                    nilai_dbi_terpilih = df_dbi[df_dbi["Jumlah Cluster (k)"] == k_terpilih]["Davies-Bouldin Index"].values[0]

                    # Tampilkan hasil DBI untuk jumlah cluster terpilih
                    st.success(f"\u2705 Davies-Bouldin Index (DBI) untuk {k_terpilih} cluster: {nilai_dbi_terpilih:.4f}")

                    # Visualisasi
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df_dbi["Jumlah Cluster (k)"], df_dbi["Davies-Bouldin Index"], 'ro-', marker='o')
                    ax.set_xlabel("Jumlah Cluster (k)")
                    ax.set_ylabel("Davies-Bouldin Index")
                    ax.set_title("Davies-Bouldin Index untuk berbagai jumlah cluster (2-10)")
                    ax.grid(True)
                    st.pyplot(fig)

                    # Tampilkan tabel
                    st.write("### Hasil Davies-Bouldin Index untuk Cluster 2 sampai 10")
                    st.write(df_dbi)

                else:
                    st.warning("Jumlah cluster terlalu sedikit untuk menghitung Davies-Bouldin Index.")

        else:
            st.warning("Data tidak memiliki kolom 'Cluster' atau kosong.")
    else:
        st.warning("Data clustering belum tersedia di sesi Streamlit.")


# if selected == "Analisa Hasil":
#     st.title("Analisa Hasil")

#     if "clustered_data" in st.session_state:
#         df_clustered = st.session_state["clustered_data"].copy()

#         if "Cluster" in df_clustered.columns and not df_clustered.empty:
#             X = df_clustered.drop(columns=["Cluster"])
#             labels = df_clustered["Cluster"]

#             # ============================
#             # Analisis Rata-Rata Karakteristik UMKM per Cluster
#             # ============================
#             st.subheader("Rata-Rata Karakteristik UMKM per Cluster")

#             if "encoded_data" in st.session_state:
#                 df_transformed = st.session_state["encoded_data"].copy()

#                 df_result_full = df_transformed.copy()
#                 df_result_full["Cluster"] = df_clustered["Cluster"].values

#                 numeric_columns_asli = df_transformed.select_dtypes(include=["int64", "float64"]).columns.tolist()
#                 cluster_means = df_result_full.groupby("Cluster")[numeric_columns_asli].mean()

#                 cluster_means_formatted = cluster_means.round(0).astype(int).applymap(lambda x: f"{x:,}".replace(",", "."))

#                 st.dataframe(cluster_means_formatted)
#             else:
#                 st.warning("\u26A0 Data transformasi belum ditemukan. Silakan lakukan preprocessing terlebih dahulu.")
#         else:
#             st.warning("\u26A0 Model KMeans belum ditemukan. Silakan jalankan proses clustering terlebih dahulu.")
#     else:
#         st.warning("\u26A0 Data clustering tidak lengkap atau belum tersedia.")


elif selected == "Analisa Hasil":
    st.title("Analisa Hasil")

    # Periksa apakah data hasil clustering tersedia
    if "clustered_data" not in st.session_state:
        st.warning("\u26A0 Data hasil clustering tidak ditemukan. Silakan jalankan proses clustering terlebih dahulu.")
        st.stop()

    df_clustered = st.session_state["clustered_data"].copy()

    if "Cluster" not in df_clustered.columns or df_clustered.empty:
        st.warning("\u26A0 Data tidak memiliki kolom 'Cluster' atau data kosong.")
        st.stop()

    # Gunakan data transformasi jika tersedia, jika tidak gunakan data asli
    if "encoded_data" in st.session_state:
        df_transformed = st.session_state["encoded_data"].copy()
    elif "uploaded_data" in st.session_state:
        df_transformed = st.session_state["uploaded_data"].copy()
    else:
        st.warning("\u26A0 Data transformasi maupun data asli tidak ditemukan. Silakan unggah atau proses data terlebih dahulu.")
        st.stop()

    # Gabungkan hasil clustering dengan data asli/transformasi
    if len(df_transformed) != len(df_clustered):
        st.warning("\u26A0Panjang data transformasi dan hasil clustering tidak sama. Tidak dapat melakukan analisa.")
        st.stop()

    df_result_full = df_transformed.copy()
    df_result_full["Cluster"] = df_clustered["Cluster"].values

    # Hitung rata-rata per cluster untuk fitur numerik
    st.subheader("Rata-Rata Karakteristik UMKM per Cluster")

    numeric_columns = df_result_full.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "Cluster" in numeric_columns:
        numeric_columns.remove("Cluster")

    cluster_means = df_result_full.groupby("Cluster")[numeric_columns].mean()
    cluster_means_formatted = cluster_means.round(0).astype(int).applymap(lambda x: f"{x:,}".replace(",", "."))

    st.dataframe(cluster_means_formatted)
