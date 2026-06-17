import os
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


st.set_page_config(
    page_title="Sistem Analisis Pola Penumpang Jakarta",
    page_icon="None",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
        /* Upload File */
    [data-testid="stFileUploader"] section {
        background-color: #FFF0F2 !important;
        border: 2px dashed #FF8E9E !important;
        border-radius: 15px !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #FF8E9E !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    

    [data-testid="stFileUploader"] button:hover {
        background-color: #FF6B81 !important;
    }
            
        /* Download Button */
    [data-testid="stDownloadButton"] button {
        background-color: #FF8E9E !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }
    
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div {
        color: #5C3D42 !important;
        font-weight: 500 !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background-color: #FF6B81 !important;
        color: white !important;
    }
    
    /* Memaksa font utama & latar belakang pink pastel super lembut di semua kontainer */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', sans-serif;
        background-color: #FFF5F6 !important;
        color: #5C3D42 !important;
    }
    
    /* Menghilangkan latar belakang header bawaan Streamlit */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* Mengubah warna teks default agar tidak hitam pekat */
    h1, h2, h3, h4, h5, h6, p, span, label, li, div {
        color: #5C3D42 !important;
    }
    
    /* Pengaturan bar sisi (Sidebar) dengan warna merah muda pastel yang manis */
    section[data-testid="stSidebar"] {
        background-color: #FFEAEF !important;
        border-right: 2px solid #FFD1DC !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #7D4F56 !important;
    }
    
    /* Mengubah warna slider Streamlit agar senada dengan pink coquette */
    .stSlider [data-baseweb="slider"] {
        background-color: #FFD1DC !important;
    }
    
    /* Kad visual premium bertemakan pastel pink */
    .metric-card {
        background-color: #FFFFFF !important;
        border: 2px solid #FFE3E8 !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 16px -4px rgba(255, 183, 197, 0.2), 0 4px 6px -2px rgba(255, 183, 197, 0.1);
        margin-bottom: 20px;
    }
    
    .metric-title {
        color: #8C7A7E !important;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #FF8E9E !important;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Desain header utama (Header) */
    .dashboard-header {
        background: linear-gradient(135deg, #FFD1DC 0%, #FFF0F2 100%) !important;
        padding: 40px;
        border-radius: 20px;
        color: #5C3D42;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px -5px rgba(255, 183, 197, 0.3);
        border: 2px solid #FFE3E8;
    }
    
    .dashboard-title {
        color: #5C3D42 !important;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 10px 0;
    }
    
    .dashboard-subtitle {
        color: #8C6A70 !important;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
        line-height: 1.5;
    }
    
    .section-title {
        color: #5C3D42 !important;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #FF8E9E;
        padding-left: 12px;
    }
    
    /* Panel informasi dan sukses khusus */
    .custom-info-card {
        background-color: #FFF0F2 !important;
        border-left: 4px solid #FF8E9E !important;
        padding: 16px 20px;
        border-radius: 12px;
        color: #8C3A47 !important;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 24px;
        border-top: 1px solid #FFE3E8;
        border-right: 1px solid #FFE3E8;
        border-bottom: 1px solid #FFE3E8;
    }
    
    .custom-success-card {
        background-color: #FFF5F6 !important;
        border: 2px solid #FFE3E8 !important;
        border-left: 4px solid #FF8E9E !important;
        padding: 20px;
        border-radius: 16px;
        color: #6E3E45 !important;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(255, 183, 197, 0.1);
    }

    /* Mengubah warna tombol agar senada */
    .stButton>button {
        background-color: #FFD1DC !important;
        color: #5C3D42 !important;
        border: 1px solid #FFE3E8 !important;
        border-radius: 20px !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FF8E9E !important;
        color: #FFFFFF !important;
        border-color: #FF8E9E !important;
        box-shadow: 0 4px 12px rgba(255, 142, 158, 0.3) !important;
    }

    /* Gaya tabel agar menyatu dengan tema */
    .stDataFrame {
        background-color: #FFFFFF !important;
        border: 1px solid #FFE3E8 !important;
        border-radius: 12px !important;
    }
    
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="dashboard-header">
    <h1 class="dashboard-title">Analisis Pola Penumpang Transportasi Jakarta</h1>
    <p class="dashboard-subtitle">Sistem segmentasi perilaku dan tren harian mobilitas komuter DKI Jakarta menggunakan K-Means Clustering berbasis integrasi multi-moda.</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("""
<div style="margin-bottom: 24px;">
    <p style="color: #FF8E9E; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.1em; margin: 0;">Sistem Analisis</p>
    <h3 style="margin: 0; font-weight: 700; color: #5C3D42; font-size: 1.25rem;">Panel Navigasi</h3>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Unggah Dataset CSV", 
    type=["csv"],
    help="Dataset wajib memiliki kolom 'tanggal', 'jenis_moda', dan 'jumlah_penumpang_per_hari'."
)

st.sidebar.markdown("<hr style='border: 0; border-top: 1px solid #FFE3E8; margin: 24px 0;'>", unsafe_allow_html=True)

st.sidebar.markdown("""
<p style="color: #6E5357; font-size: 0.85rem; font-weight: 600; margin-bottom: 8px;">Parameter Pengelompokan</p>
""", unsafe_allow_html=True)

k = st.sidebar.slider(
    "Jumlah Cluster (K)",
    min_value=2,
    max_value=9,
    value=3,
    help="Tentukan jumlah cluster harian yang ingin dihasilkan oleh model."
)

# Palet warna untuk visualisasi data
COLOR_PALETTE = [
    '#FFC2D1', 
    '#FF85A1',  
    '#B5179E',
    '#7B2CBF',
    '#4361EE',
    '#4CC9F0',
    '#38B000',
    '#FF8800',
    '#E63946'
]

# ENJIN RUNTIME (DATA SEDIA ADA vs DATA KOSONG)
if not uploaded_file:
    # Skrin selamat datang premium
    st.markdown("""
    <div class="custom-info-card" style="background-color: #FFF9FA; border-left-color: #FFB7C5; color: #6E5357;">
        Sebelum memulai proses segmentasi otomatis, silakan unggah berkas komparasi data perjalanan Anda melalui panel di sebelah kiri.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Panduan Struktur Dataset"):
        st.markdown("""
        <p style="color: #6E5357; font-size: 0.9rem; line-height: 1.6; margin-bottom: 20px;">
            Aplikasi mengonversi bentuk data transaksi harian Anda menjadi model pivot deret waktu berdasarkan tanggal perjalanan. Pastikan dokumen Anda memiliki konfigurasi kolom berikut:
        </p>
        """, unsafe_allow_html=True)
        
        example_data = pd.DataFrame({
            'tanggal': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-02'],
            'jenis_moda': ['MRT', 'LRT', 'Transjakarta', 'MRT', 'LRT', 'Transjakarta'],
            'jumlah_penumpang_per_hari': [85000, 12000, 240000, 92000, 14000, 255000]
        })
        st.table(example_data)

else:
    
    # PAIP DATA BAHAGIAN 1: MUAT & TRANSFORMASI
    try:
        df = pd.read_csv(uploaded_file)
        
        # Mengesahkan lajur wajib ada sebelum diproses
        required_cols = {'tanggal', 'jenis_moda', 'jumlah_penumpang_per_hari'}
        if not required_cols.issubset(df.columns):
            st.error("Format data salah. Dataset harus memuat kolom: 'tanggal', 'jenis_moda', dan 'jumlah_penumpang_per_hari'.")
            st.stop()
            
        # Menukar format tarikh
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        
        # Membina jadual pivot
        df_pivot = df.pivot_table(
            index='tanggal',
            columns='jenis_moda',
            values='jumlah_penumpang_per_hari'
        )
        
        # Menggantikan nilai yang hilang dengan purata moda masing-masing
        X = df_pivot.fillna(df_pivot.mean())
        
        # Penskalaan Data (Scaling)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {str(e)}")
        st.stop()

    # Membina susun atur tab utama
    tab1, tab2, tab3 = st.tabs(["Eksplorasi & Transformasi", "Optimasi & Evaluasi Model", "Karakteristik & Analisis Solusi"])

    # --- TAB 1: EKSPLORASI DATA & PIVOT ---
    with tab1:
        st.markdown('<h3 class="section-title">Hasil Transformasi Matriks (Data Pivot)</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #6E5357; font-size: 0.9rem; line-height: 1.5; margin-bottom: 20px;">
            Berikut adalah representasi data harian setelah dikelompokkan berdasarkan tanggal sebagai indeks dan jenis transportasi publik sebagai pemboleh ubah kolom.
        </p>
        """, unsafe_allow_html=True)
        
        st.dataframe(X, use_container_width=True, height=350)
        
        # Grid Statistik Ringkas
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Jumlah Hari Diperhati</div>
                <div class="metric-value">{X.shape[0]:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_info2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Jumlah Variasi Moda</div>
                <div class="metric-value">{X.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_info3:
            raw_total_rows = df.shape[0]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Jumlah Baris Input Asal</div>
                <div class="metric-value">{raw_total_rows:,}</div>
            </div>
            """, unsafe_allow_html=True)

    
    # PAIP DATA BAHAGIAN 2: PENGIRAAN MODEL
    # Pelaksanaan model utama K-Means
    X['Total_Harian'] = X.sum(axis=1)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_labels = model.fit_predict(X_scaled)
    X['cluster'] = raw_labels

    # REMAPPING TRICK: Mengurutkan ID cluster berdasarkan rata-rata Total_Harian
    # 1. Gabungkan dulu cluster ke data transaksi asli untuk menghitung total harian riil
    df_temp = df.merge(X['cluster'], left_on='tanggal', right_index=True, how='left')
    
    # 2. Hitung total harian murni per tanggal, lalu cari rata-rata per cluster-nya
    total_riil_per_cluster = (
        df_temp.groupby(['tanggal', 'cluster'])['jumlah_penumpang_per_hari']
        .sum()
        .groupby('cluster')
        .mean()
    )
    
    # 3. Urutkan ID cluster dari yang rata-rata total penumpangnya paling sedikit ke paling banyak
    idx_terurut = total_riil_per_cluster.sort_values().index
    mapping_id = {id_lama: id_baru for id_baru, id_lama in enumerate(idx_terurut)}
    
    # 4. Terapkan ID cluster baru yang sudah pasti urut murni (0 = Sepi, 1 = Sedang, 2 = Ramai)
    X['cluster'] = X['cluster'].map(mapping_id)
    labels = X['cluster'].to_numpy() # Sinkronisasi label baru untuk PCA

    # Pengiraan Metrik Penilaian
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)

    # DEFINISIKAN NAMA CLUSTER DI SINI (Supaya TAB 2 dan TAB 3 bisa membacanya tanpa error)
    nama_cluster = {}
    for i in range(k):
        if i == 0:
            nama_cluster[i] = "Hari Sepi"
        elif i == k - 1:
            nama_cluster[i] = "Hari Ramai"
        else:
            nama_cluster[i] = f"Hari Sedang {i}" if k > 3 else "Hari Sedang"

    X['nama_kategori'] = X['cluster'].map(nama_cluster)

    # --- TAB 2: OPTIMASI & PENILAIAN MODEL ---
    with tab2:
        col_opt1, col_opt2 = st.columns([4, 6], gap="large")
        
        with col_opt1:
            st.markdown('<h3 class="section-title">Analisis Titik Siku (Elbow Method)</h3>', unsafe_allow_html=True)
            
            # Pengiraan inersia bagi kaedah Elbow
            inertia = []
            K_range = range(1, 10)
            for i in K_range:
                km = KMeans(n_clusters=i, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertia.append(km.inertia_)
            
            # Penggayaan Eksklusif Matplotlib (Sesuai dengan latar belakang pink pastel)
            fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4.5), facecolor='#FFF5F6')
            ax_elbow.set_facecolor('#FFF5F6')
            ax_elbow.plot(K_range, inertia, marker='o', markersize=6, linewidth=2, color='#FF8E9E', markerfacecolor='#FF8E9E')
            ax_elbow.set_xlabel('Jumlah Cluster (K)', fontsize=9, color='#6E5357', fontweight='semibold')
            ax_elbow.set_ylabel('Inertia (Jumlah Kuadrat Kesalahan)', fontsize=9, color='#6E5357', fontweight='semibold')
            ax_elbow.tick_params(colors='#8C6A70', labelsize=8)
            ax_elbow.grid(color='#FFE3E8', linestyle='--', linewidth=0.5)
            ax_elbow.spines['top'].set_visible(False)
            ax_elbow.spines['right'].set_visible(False)
            ax_elbow.spines['left'].set_color('#FFD1DC')
            ax_elbow.spines['bottom'].set_color('#FFD1DC')
            
            st.pyplot(fig_elbow, clear_figure=True)
            
        with col_opt2:
            st.markdown('<h3 class="section-title">Metrik Kuantitatif K-Means</h3>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Skor Silhouette</div>
                    <div class="metric-value">{round(sil, 3)}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Davies-Bouldin (DBI)</div>
                    <div class="metric-value">{round(dbi, 3)}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Calinski-Harabasz</div>
                    <div class="metric-value">{round(ch, 1)}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("""
            <div style="background-color: #FFF9FA; padding: 20px; border-radius: 12px; border: 2px solid #FFE3E8; font-size: 0.9rem; color: #6E5357; line-height: 1.5;">
                <strong>Panduan Interpretasi Metrik:</strong><br>
                1. <strong>Skor Silhouette:</strong> Bernilai antara -1 hingga 1. Nilai yang mendekati 1 menunjukkan pemisahan kelompok yang sangat baik.<br>
                2. <strong>Indeks Davies-Bouldin:</strong> Nilai yang lebih kecil adalah lebih baik, menunjukkan pemisahan kelompok yang optimum.<br>
                3. <strong>Skor Calinski-Harabasz:</strong> Skor yang lebih tinggi menggambarkan struktur kelompok yang lebih padat dan jelas.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border: 0; border-top: 1px solid #FFE3E8; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Visualisasi Reduksi Dimensi 2D PCA</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #6E5357; font-size: 0.9rem; line-height: 1.5; margin-bottom: 20px;">
            Untuk memetakan data berdimensi banyak ke dalam koordinat 2D, analisis komponen utama (PCA) digunakan guna mereduksi kompleksitas fitur tanpa kehilangan informasi variasi yang penting.
        </p>
        """, unsafe_allow_html=True)

        # Menjalankan unjuran PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X_scaled)
        
        df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        df_pca['Kategori'] = labels
        df_pca['Kategori'] = df_pca['Kategori'].map(nama_cluster)
        df_pca['Tanggal'] = X.index.strftime('%Y-%m-%d')
        
        # Carta Sebaran Interaktif Plotly 
        fig_pca = px.scatter(
            df_pca,
            x='PC1',
            y='PC2',
            color='Kategori',
            category_orders={"Kategori": ["Hari Sepi", "Hari Sedang", "Hari Ramai"]},
            hover_data=['Tanggal'],
            title="Taburan Titik Harian Hasil Reduksi Dimensi PCA",
            template="plotly_white",
            color_discrete_map={
                "Hari Sepi": COLOR_PALETTE[0],
                "Hari Sedang": COLOR_PALETTE[1],
                "Hari Ramai": COLOR_PALETTE[2] if k == 3 else COLOR_PALETTE[-1]
            }
        )
        
        
        # TAMBAHKAN KODE BARU INI UNTUK GARIS PINGGIR BULATAN PCA
        fig_pca.update_traces(
            marker=dict(
                size=10,                     
                line=dict(
                    width=1,                  
                    color='#5C3D42'           
                )
            )
        )
        

        fig_pca.update_layout(
            font_family="Inter",
            font=dict(
                color="#5C3D42",
                size=12
            ),
            title_font_size=16,
            title_font_color="#5C3D42",
            legend_title_font_color="#5C3D42",
            legend_font_color="#5C3D42",
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor="#FFF5F6",
            plot_bgcolor="#FFFFFF"
        )
        fig_pca.update_xaxes(
            title_font_color="#5C3D42",
            tickfont=dict(color="#5C3D42"),
            showgrid=True,
            gridcolor='#FFE3E8'
        )

        fig_pca.update_yaxes(
            title_font_color="#5C3D42",
            tickfont=dict(color="#5C3D42"),
            showgrid=True,
            gridcolor="#FDC5CF"
        )
        st.plotly_chart(fig_pca, use_container_width=True)


    
    # PAIP DATA BAHAGIAN 3: KATEGORI & INSAIT
    # Menggabungkan kluster harian semula untuk ringkasan agregat
    df_cluster = df.merge(
        X['cluster'],
        left_on='tanggal',
        right_index=True,
        how='left'
    )

    # Mengira statistik kluster berdasarkan total harian asli yang sudah urut konsisten
    cluster_sort = X.groupby('cluster')['Total_Harian'].agg(['mean', 'min', 'max'])
    cluster_sort.columns = ['Rata-rata', 'Minimum', 'Maksimum']
    cluster_sort['Kategori'] = cluster_sort.index.map(nama_cluster)
   

    # --- TAB 3: CIRI-CIRI & PENYELESAIAN ---
    with tab3:
        st.markdown('<h3 class="section-title">Segmentasi Karakteristik Kelompok</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #6E5357; font-size: 0.9rem; line-height: 1.5; margin-bottom: 20px;">
            Berikut merupakan rincian statistik perjalanan harian kumulatif yang dipetakan ke dalam beberapa kategori intensitas mobilitas harian di Jakarta.
        </p>
        """, unsafe_allow_html=True)

        col_tbl1, col_tbl2 = st.columns([6, 4], gap="large")
        
        with col_tbl1:
            st.markdown("<p style='font-weight:600; color:#5C3D42; margin-bottom:10px;'>Statistik Agregat Harian</p>", unsafe_allow_html=True)
            st.dataframe(cluster_sort[['Kategori', 'Rata-rata', 'Minimum', 'Maksimum']].round(0).set_index('Kategori'), use_container_width=True)
            
            st.markdown("<p style='font-weight:600; color:#5C3D42; margin-top:25px; margin-bottom:10px;'>Rata-rata Distribusi Penumpang Menurut Moda Transportasi</p>", unsafe_allow_html=True)
            
            # ==============================================================================
            # SOLUSI TOTAL: HITUNG DISTRIBUSI MODA DARI DATA AGREGAT ASLI
            # ==============================================================================
            # 1. Ambil dataframe asli hasil merge klaster harian (df_cluster)
            # 2. Kelompokkan berdasarkan 'cluster' (ID angka yang sudah urut konsisten)
            # 3. Ambil rata-ratanya untuk tiap jenis moda
            df_moda_urut = df_cluster.groupby(['cluster', 'jenis_moda'])['jumlah_penumpang_per_hari'].mean().unstack().fillna(0)
            
            # 4. Ubah indeks angka (0, 1, 2) menjadi nama kategori teks yang sesuai
            df_moda_urut.index = df_moda_urut.index.map(nama_cluster)
            
            # 5. Paksa urutan baris secara manual agar mutlak dari Sepi -> Sedang -> Ramai
            urutan_tetap = ["Hari Sepi", "Hari Sedang", "Hari Ramai"]
            urutan_valid = [k for k in urutan_tetap if k in df_moda_urut.index]
            hasil_moda = df_moda_urut.reindex(urutan_valid)
            # ==============================================================================
            
            st.dataframe(hasil_moda.round(0), use_container_width=True)

        with col_tbl2:
            st.markdown("<p style='font-weight:600; color:#5C3D42; margin-bottom:10px;'>Carta Taburan Volume Penumpang Kategori</p>", unsafe_allow_html=True)
            
            fig_bar = px.bar(
                cluster_sort,
                x='Kategori',
                y='Rata-rata',
                text='Rata-rata',
                template="plotly_white",
                color='Kategori',
                color_discrete_sequence=COLOR_PALETTE[:k]
            )
            fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_bar.update_layout(
                font_family="Inter",
                font=dict(
                    color="#5C3D42",
                    size=12
                ),
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                height=320,
                paper_bgcolor="#FFF5F6",
                plot_bgcolor="#FFFFFF"
            )

            fig_bar.update_xaxes(
                tickfont=dict(color="#5C3D42"),
                title_font_color="#5C3D42"
            )

            fig_bar.update_yaxes(
                tickfont=dict(color="#5C3D42"),
                title_font_color="#5C3D42",
                showgrid=True,
                gridcolor='#FFD1DC'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Paparan kad interpretasi hasil segmentasi
        st.markdown("<hr style='border: 0; border-top: 1px solid #FFE3E8; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Interpretasi Hasil Segmentasi</h3>', unsafe_allow_html=True)
        
        for idx, row in cluster_sort.iterrows():
            st.markdown(f"""
            <div class="custom-success-card">
                <h4 style="margin: 0 0 8px 0; font-size: 1.1rem; font-weight: 700; color: #8C3A47;">{row['Kategori']}</h4>
                <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: #8C535C;">
                    Menyumbang rata-rata harian sebanyak <strong>{int(row['Rata-rata']):,}</strong> penumpang. 
                    Batas pergerakan dicatatkan pada rentang minimum harian sebanyak <strong>{int(row['Minimum']):,}</strong> penumpang hingga puncak kapasitas harian sebanyak <strong>{int(row['Maksimum']):,}</strong> penumpang.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Algoritma pengesyoran tindakan secara dinamik
        st.markdown('<h3 class="section-title">Rekomendasi Manajerial & Taktis</h3>', unsafe_allow_html=True)
        try:
            ramai_data = cluster_sort[cluster_sort['Kategori'] == "Hari Ramai"]
            if not ramai_data.empty:
                val_rata_ramai = ramai_data['Rata-rata'].values[0]
                st.markdown(f"""
                <div class="custom-info-card">
                    <strong>Rekomendasi Operasional Khusus (Hari Ramai):</strong><br>
                    Hari dalam kategori ramai ini mencatatkan rata-rata penumpang sebanyak <strong>{int(val_rata_ramai):,}</strong> sehari. Strategi mitigasi taktis yang direkomendasikan:<br>
                    • Menambah gerbong atau unit armada cadangan pada jam puncak kemacetan.<br>
                    • Meningkatkan frekuensi perjalanan (memperkecil selang waktu headway).<br>
                    • Mengoptimalkan sistem pengelolaan antrean di stasiun dan halte guna menghindari penumpukan komuter.<br>
                    • Menempatkan personel keamanan tambahan di lokasi transit tumpuan ramai komuter.
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass

        # Jadual penjejakan harian
        st.markdown("<hr style='border: 0; border-top: 1px solid #FFE3E8; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Keputusan Klasifikasi Harian</h3>', unsafe_allow_html=True)
        
        hasil_hari = pd.DataFrame({
            'Tanggal': X.index.strftime('%Y-%m-%d'),
            'ID Cluster': labels
        })
        hasil_hari['Kategori'] = hasil_hari['ID Cluster'].map(nama_cluster)
        
        st.dataframe(hasil_hari.set_index('Tanggal'), use_container_width=True, height=300)

        # Modul muat turun data
        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
        csv_data = hasil_hari.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Berkas Hasil Segmentasi CSV",
            data=csv_data,
            file_name="hasil_analisis_segmentasi_jakarta.csv",
            mime="text/csv",
            use_container_width=False
        )
        st.markdown("</div>", unsafe_allow_html=True)


# FOOTER
st.markdown("<hr style='border: 0; border-top: 1px solid #FFE3E8; margin: 60px 0 24px 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #B39296; font-size: 0.8rem; margin-bottom: 24px; letter-spacing: 0.05em;">
    SISTEM PORTAL ANALISIS K-MEANS • DINAS PERHUBUNGAN DKI JAKARTA • TRANSMISI DATA AMAN
</div>
""", unsafe_allow_html=True)