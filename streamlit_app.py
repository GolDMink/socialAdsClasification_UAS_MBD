import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Prediksi Iklan Jejaring Sosial",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-result {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .will-buy {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .will-not-buy {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model_objects():
    """Load semua objek model dan mengembalikannya"""
    try:
        model = joblib.load('best_social_ads_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        feature_info = joblib.load('feature_info.pkl')
        model_metrics = joblib.load('model_metrics.pkl')
        return model, scaler, label_encoder, feature_info, model_metrics
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        st.error("Silakan jalankan notebook Jupyter terlebih dahulu untuk menghasilkan file model.")
        return None, None, None, None, None

# Load dataset
@st.cache_data
def load_dataset():
    """Memuat dataset"""
    try:
        df = pd.read_csv('Social_Network_Ads.csv')
        return df
    except FileNotFoundError:
        st.error("File dataset 'Social_Network_Ads.csv' tidak ditemukan!")
        return None

def main():
    # Main title
    st.markdown('<h1 class="main-header">🛍️ Prediksi Pembelian Iklan Jejaring Sosial</h1>', unsafe_allow_html=True)
    
    # Load model objects
    model, scaler, label_encoder, feature_info, model_metrics = load_model_objects()
    
    if model is None:
        st.stop()
    
    # Load dataset
    df = load_dataset()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔧 Navigasi")
    page = st.sidebar.selectbox(
        "Pilih halaman:",
        ["🏠 Beranda", "📊 Ikhtisar Dataset", "🔮 Buat Prediksi", "📈 Performa Model", "📋 Tentang"]
    )
    
    if page == "🏠 Beranda":
        show_home_page(df, model_metrics)
    elif page == "📊 Ikhtisar Dataset":
        show_dataset_overview(df)
    elif page == "🔮 Buat Prediksi":
        show_prediction_page(model, scaler, label_encoder, feature_info)
    elif page == "📈 Performa Model":
        show_model_performance(model_metrics, df)
    elif page == "📋 Tentang":
        show_about_page()

def show_home_page(df, model_metrics):
    """Tampilkan halaman beranda dengan ikhtisar"""
    st.markdown('<h2 class="sub-header">🎯 Ikhtisar Proyek</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Info Dataset</h3>
            <p><strong>Total Data:</strong> {}</p>
            <p><strong>Fitur:</strong> Usia, Gaji, Jenis Kelamin</p>
            <p><strong>Target:</strong> Keputusan Pembelian</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Info Model</h3>
            <p><strong>Algoritma:</strong> Random Forest</p>
            <p><strong>Akurasi:</strong> {:.2f}%</p>
            <p><strong>Skor AUC:</strong> {:.3f}</p>
        </div>
        """.format(
            model_metrics['accuracy'] * 100,
            model_metrics['auc_score']
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Kasus Penggunaan</h3>
            <p>• Penargetan Kampanye Marketing</p>
            <p>• Segmentasi Pelanggan</p>
            <p>• Prediksi Penjualan</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown('<h2 class="sub-header">📊 Statistik Cepat</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    purchase_rate = df['Purchased'].mean()
    avg_age = df['Age'].mean()
    avg_salary = df['EstimatedSalary'].mean()
    male_ratio = (df['Gender'] == 'Male').mean()
    
    with col1:
        st.metric("Tingkat Pembelian", f"{purchase_rate:.1%}")
    with col2:
        st.metric("Rata-rata Usia", f"{avg_age:.1f} tahun")
    with col3:
        st.metric("Rata-rata Gaji", f"${avg_salary:,.0f}")
    with col4:
        st.metric("Rasio Pria", f"{male_ratio:.1%}")
    
    # Sample data
    st.markdown('<h2 class="sub-header">🔍 Contoh Data</h2>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

def show_dataset_overview(df):
    """Tampilkan ikhtisar dataset dan visualisasi"""
    st.markdown('<h2 class="sub-header">📊 Ikhtisar Dataset</h2>', unsafe_allow_html=True)
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Bentuk Dataset:**", df.shape)
        st.write("**Kolom:**", list(df.columns))
        st.write("**Tipe Data:**")
        st.write(df.dtypes)
    
    with col2:
        st.write("**Nilai yang Hilang:**")
        missing_values = df.isnull().sum()
        st.write(missing_values)
        
        st.write("**Ringkasan Statistik:**")
        st.write(df.describe())
    
    # Visualizations
    st.markdown('<h2 class="sub-header">📈 Visualisasi Data</h2>', unsafe_allow_html=True)
    
    # Target distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_target = px.pie(
            values=df['Purchased'].value_counts().values,
            names=['Tidak Membeli', 'Membeli'],
            title="Distribusi Pembelian"
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col2:
        fig_gender = px.pie(
            values=df['Gender'].value_counts().values,
            names=['Pria' if x == 'Male' else 'Wanita' for x in df['Gender'].value_counts().index],
            title="Distribusi Jenis Kelamin"
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Age and Salary distributions
    col1, col2 = st.columns(2)
    
    with col1:
        fig_age = px.histogram(
            df, x='Age', nbins=20,
            title="Distribusi Usia",
            labels={'Age': 'Usia', 'count': 'Frekuensi'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        fig_salary = px.histogram(
            df, x='EstimatedSalary', nbins=20,
            title="Distribusi Gaji",
            labels={'EstimatedSalary': 'Estimasi Gaji', 'count': 'Frekuensi'}
        )
        st.plotly_chart(fig_salary, use_container_width=True)
    
    # Correlation with target
    st.markdown('<h3>🔗 Hubungan dengan Keputusan Pembelian</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_age_purchase = px.box(
            df, x='Purchased', y='Age',
            title="Usia vs Keputusan Pembelian",
            labels={'Purchased': 'Keputusan Pembelian', 'Age': 'Usia'}
        )
        st.plotly_chart(fig_age_purchase, use_container_width=True)
    
    with col2:
        fig_salary_purchase = px.box(
            df, x='Purchased', y='EstimatedSalary',
            title="Gaji vs Keputusan Pembelian",
            labels={'Purchased': 'Keputusan Pembelian', 'EstimatedSalary': 'Estimasi Gaji'}
        )
        st.plotly_chart(fig_salary_purchase, use_container_width=True)
    
    # Scatter plot
    fig_scatter = px.scatter(
        df, x='Age', y='EstimatedSalary', 
        color='Purchased',
        title="Usia vs Gaji (Warna berdasarkan Pembelian)",
        labels={'Age': 'Usia', 'EstimatedSalary': 'Estimasi Gaji', 'Purchased': 'Keputusan Pembelian'},
        color_discrete_sequence=['red', 'green'],
        category_orders={'Purchased': [0, 1]},
        hover_data=['Gender']
    )
    fig_scatter.update_layout(
        showlegend=True,
        legend_title_text="Keputusan Pembelian",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

def show_prediction_page(model, scaler, label_encoder, feature_info):
    """Tampilkan halaman prediksi"""
    st.markdown('<h2 class="sub-header">🔮 Buat Prediksi Pembelian</h2>', unsafe_allow_html=True)
    
    # Input form
    st.markdown("### Masukkan Informasi Pelanggan:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Usia", min_value=18, max_value=70, value=30, step=1)
    
    with col2:
        salary = st.slider("Estimasi Gaji ($)", min_value=15000, max_value=200000, value=50000, step=1000)
    
    with col3:
        gender = st.selectbox("Jenis Kelamin", options=['Pria', 'Wanita'])
    
    # Map gender back to English for model
    gender_map = {'Pria': 'Male', 'Wanita': 'Female'}
    gender_eng = gender_map[gender]
    
    # Prediction button
    if st.button("🔮 Prediksi Keputusan Pembelian", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'EstimatedSalary': [salary],
            'Gender': [gender_eng]
        })
        
        # Preprocess input
        input_data['Gender_encoded'] = label_encoder.transform(input_data['Gender'])
        input_processed = input_data[['Age', 'EstimatedSalary', 'Gender_encoded']]
        
        # Scale input
        input_scaled = scaler.transform(input_processed)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Display result
        st.markdown("### 🎯 Hasil Prediksi:")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-result will-buy">
                ✅ AKAN MEMBELI
                <br>
                Probabilitas: {prediction_proba[1]:.1%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result will-not-buy">
                ❌ TIDAK AKAN MEMBELI
                <br>
                Probabilitas: {prediction_proba[0]:.1%}
            </div>
            """, unsafe_allow_html=True)
        
        # Show probabilities
        st.markdown("### 📊 Probabilitas Prediksi:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tidak Akan Membeli", f"{prediction_proba[0]:.1%}")
        
        with col2:
            st.metric("Akan Membeli", f"{prediction_proba[1]:.1%}")
        
        # Probability chart
        fig_prob = go.Figure(data=[
            go.Bar(
                x=['Tidak Akan Membeli', 'Akan Membeli'],
                y=[prediction_proba[0], prediction_proba[1]],
                marker_color=['red', 'green']
            )
        ])
        fig_prob.update_layout(
            title="Probabilitas Prediksi",
            yaxis_title="Probabilitas",
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Feature importance for this prediction
        st.markdown("### 🔍 Kontribusi Fitur:")
        
        feature_names = ['Usia', 'Estimasi Gaji', 'Jenis Kelamin']
        feature_values = [age, salary, gender]
        
        importance_df = pd.DataFrame({
            'Fitur': feature_names,
            'Nilai': feature_values,
            'Kepentingan': model.feature_importances_
        })
        
        fig_importance = px.bar(
            importance_df, x='Fitur', y='Kepentingan',
            title="Kepentingan Fitur dalam Model",
            labels={'Kepentingan': 'Skor Kepentingan'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Batch prediction
    st.markdown("### 📁 Prediksi Batch")
    st.markdown("Unggah file CSV dengan kolom: Age, EstimatedSalary, Gender")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_columns = ['Age', 'EstimatedSalary', 'Gender']
            if all(col in batch_df.columns for col in required_columns):
                # Preprocess batch data
                batch_df['Gender_encoded'] = label_encoder.transform(batch_df['Gender'])
                batch_processed = batch_df[['Age', 'EstimatedSalary', 'Gender_encoded']]
                
                # Scale batch data
                batch_scaled = scaler.transform(batch_processed)
                
                # Make predictions
                batch_predictions = model.predict(batch_scaled)
                batch_probabilities = model.predict_proba(batch_scaled)
                
                # Add results to dataframe
                batch_df['Prediksi'] = batch_predictions
                batch_df['Probabilitas_Pembelian'] = batch_probabilities[:, 1]
                batch_df['Keputusan'] = batch_df['Prediksi'].map({0: 'Tidak Akan Membeli', 1: 'Akan Membeli'})
                
                # Display results
                st.markdown("### 📊 Hasil Prediksi Batch:")
                st.dataframe(batch_df, use_container_width=True)
                
                # Summary statistics
                purchase_rate = batch_df['Prediksi'].mean()
                st.metric("Tingkat Prediksi Pembelian", f"{purchase_rate:.1%}")
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Unduh Hasil",
                    data=csv,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )
                
            else:
                st.error(f"Pastikan CSV Anda memiliki kolom: {required_columns}")
                
        except Exception as e:
            st.error(f"Error memproses file: {e}")

def show_model_performance(model_metrics, df):
    """Tampilkan metrik performa model"""
    st.markdown('<h2 class="sub-header">📈 Performa Model</h2>', unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Akurasi", f"{model_metrics.get('accuracy', 0):.1%}")
    
    with col2:
        st.metric("Skor AUC", f"{model_metrics.get('auc_score', 0):.3f}")
    
    with col3:
        # Get precision for positive class (1)
        if 'classification_report' in model_metrics:
            report = model_metrics['classification_report']
            # Try different possible keys for positive class
            for key in ['1', 1, 'positive', 'pos']:
                if key in report:
                    precision = report[key].get('precision', 0)
                    break
            else:
                precision = 0
        else:
            precision = 0
        st.metric("Presisi", f"{precision:.3f}")
    
    with col4:
        # Get recall for positive class (1)
        if 'classification_report' in model_metrics:
            report = model_metrics['classification_report']
            # Try different possible keys for positive class
            for key in ['1', 1, 'positive', 'pos']:
                if key in report:
                    recall = report[key].get('recall', 0)
                    break
            else:
                recall = 0
        else:
            recall = 0
        st.metric("Recall", f"{recall:.3f}")
    
    # Confusion Matrix
    st.markdown("### 🔍 Matriks Konfusi")
    
    if 'confusion_matrix' in model_metrics:
        cm = np.array(model_metrics['confusion_matrix'])
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
            x=['Tidak Membeli', 'Membeli'],
            y=['Tidak Membeli', 'Membeli'],
            title="Matriks Konfusi",
            text_auto=True
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Feature Importance
    st.markdown("### 🎯 Kepentingan Fitur")
    
    if 'feature_importance' in model_metrics:
        feature_names = ['Usia', 'Estimasi Gaji', 'Jenis Kelamin']
        importance_values = list(model_metrics['feature_importance'].values())
        
        fig_importance = px.bar(
            x=feature_names,
            y=importance_values,
            title="Kepentingan Fitur",
            labels={'x': 'Fitur', 'y': 'Skor Kepentingan'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Classification Report
    st.markdown("### 📊 Laporan Klasifikasi Detail")
    
    if 'classification_report' in model_metrics:
        try:
            report_df = pd.DataFrame(model_metrics['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error menampilkan laporan klasifikasi: {e}")
    
    # Cross-validation scores
    st.markdown("### 🔄 Performa Validasi Silang")
    
    cv_scores = model_metrics.get('cv_scores', [])
    if cv_scores:
        fig_cv = px.bar(
            x=[f'Lipatan {i+1}' for i in range(len(cv_scores))],
            y=cv_scores,
            title="Skor Validasi Silang",
            labels={'x': 'Lipatan', 'y': 'Skor Akurasi'}
        )
        st.plotly_chart(fig_cv, use_container_width=True)
        
        st.metric("Rata-rata Skor CV", f"{np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

def show_about_page():
    """Tampilkan halaman tentang"""
    st.markdown('<h2 class="sub-header">📋 Tentang Proyek Ini</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Ikhtisar Proyek
    
    Prediksi Pembelian Iklan Jejaring Sosial ini adalah aplikasi pembelajaran mesin yang memprediksi apakah pelanggan akan membeli produk berdasarkan informasi demografis mereka.
    
    ## 📊 Dataset
    
    Dataset berisi informasi tentang pelanggan termasuk:
    - **Usia**: Usia pelanggan
    - **Estimasi Gaji**: Perkiraan gaji tahunan pelanggan
    - **Jenis Kelamin**: Jenis kelamin pelanggan (Pria/Wanita)
    - **Pembelian**: Apakah pelanggan membeli produk (variabel target)
    
    ## 🤖 Model
    
    - **Algoritma**: Random Forest Classifier
    - **Fitur**: Usia, Estimasi Gaji, Jenis Kelamin (dikodekan)
    - **Preprocessing**: StandardScaler untuk normalisasi fitur
    - **Evaluasi**: Validasi silang dengan metrik akurasi, presisi, recall, dan AUC
    
    ## 🚀 Fitur
    
    1. **Prediksi Individual**: Prediksi keputusan pembelian untuk satu pelanggan
    2. **Prediksi Batch**: Unggah file CSV untuk prediksi multiple
    3. **Visualisasi Data**: Grafik dan diagram interaktif
    4. **Performa Model**: Metrik dan analisis performa detail
    5. **Kepentingan Fitur**: Memahami fitur mana yang paling berpengaruh
    
    ## 🛠️ Teknologi yang Digunakan
    
    - **Frontend**: Streamlit
    - **Machine Learning**: Scikit-learn
    - **Pemrosesan Data**: Pandas, NumPy
    - **Visualisasi**: Plotly, Matplotlib, Seaborn
    - **Penyimpanan Model**: Joblib
    
    ## 📈 Aplikasi Bisnis
    
    - **Penargetan Kampanye Marketing**: Identifikasi pelanggan yang paling mungkin membeli
    - **Segmentasi Pelanggan**: Kelompokkan pelanggan berdasarkan probabilitas pembelian
    - **Peramalan Penjualan**: Prediksi penjualan berdasarkan demografi pelanggan
    - **Alokasi Sumber Daya**: Optimasi pengeluaran marketing pada pelanggan berprospek tinggi
    
    ## 🔮 Pengembangan Masa Depan
    
    - Tambah fitur demografis
    - Implementasi metode ensemble
    - Pelatihan ulang model secara real-time
    - Kerangka kerja A/B testing
    - Segmentasi pelanggan lanjutan
    
    ## 📞 Kontak
    
    Untuk pertanyaan atau saran tentang proyek ini, silakan hubungi tim pengembang.
    
    ---
    
    **Dibuat dengan ❤️ menggunakan Streamlit dan Scikit-learn**
    """)

if __name__ == "__main__":
    main() 