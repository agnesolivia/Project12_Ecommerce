"""
app.py
Dashboard Streamlit untuk Project 12: Prediksi Perilaku Pembelian Pengunjung E-Commerce
Mencakup Tahap 6 (Deployment) & Tahap 7 (Monitoring) dari Machine Learning Workflow.
"""
import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from prepare_models import build_and_save_artifacts, SCALE_COLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "prediction_log.csv")

FEATURE_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "OperatingSystems", "Browser", "Region", "TrafficType",
    "Weekend",
    "Month_Aug", "Month_Dec", "Month_Feb", "Month_Jul", "Month_June",
    "Month_Mar", "Month_May", "Month_Nov", "Month_Oct", "Month_Sep",
    "VisitorType_New_Visitor", "VisitorType_Other", "VisitorType_Returning_Visitor",
]

MONTH_OPTIONS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_OPTIONS = ["Returning_Visitor", "New_Visitor", "Other"]

PRESETS = {
    "Window Shopping (Tidak Beli)": {
        "Administrative": 0, "Administrative_Duration": 0.0,
        "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 6, "ProductRelated_Duration": 120.0,
        "BounceRates": 0.05, "ExitRates": 0.08,
        "PageValues": 0.0, "SpecialDay": 0.0,
        "Month": "May", "VisitorType": "Returning_Visitor", "Weekend": False,
        "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
    },
    "Calon Pembeli Potensial": {
        "Administrative": 3, "Administrative_Duration": 85.0,
        "Informational": 1, "Informational_Duration": 30.0,
        "ProductRelated": 32, "ProductRelated_Duration": 1350.0,
        "BounceRates": 0.004, "ExitRates": 0.015,
        "PageValues": 22.5, "SpecialDay": 0.0,
        "Month": "Nov", "VisitorType": "New_Visitor", "Weekend": True,
        "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
    },
    "Pembeli Sangat Aktif (High PageValue)": {
        "Administrative": 5, "Administrative_Duration": 180.0,
        "Informational": 2, "Informational_Duration": 95.0,
        "ProductRelated": 55, "ProductRelated_Duration": 2400.0,
        "BounceRates": 0.0, "ExitRates": 0.008,
        "PageValues": 58.0, "SpecialDay": 0.0,
        "Month": "Nov", "VisitorType": "Returning_Visitor", "Weekend": False,
        "OperatingSystems": 2, "Browser": 2, "Region": 3, "TrafficType": 2,
    },
}

st.set_page_config(
    page_title="Project 12 - Prediksi Pembelian E-Commerce",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    rf_path = os.path.join(OUTPUT_DIR, "rf_model.pkl")
    knn_path = os.path.join(OUTPUT_DIR, "knn_model.pkl")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    if not (os.path.exists(rf_path) and os.path.exists(knn_path) and os.path.exists(scaler_path)):
        return build_and_save_artifacts()
    return joblib.load(rf_path), joblib.load(knn_path), joblib.load(scaler_path)


@st.cache_data
def load_reference_stats():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["Revenue"]
    with open(os.path.join(OUTPUT_DIR, "hasil_knn.json"), encoding="utf-8") as f:
        hasil_knn = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "hasil_rf.json"), encoding="utf-8") as f:
        hasil_rf = json.load(f)
    return {
        "buy_rate_train": float(y_train.mean()),
        "mean_page_values": float(X_train["PageValues"].mean()),
        "mean_exit_rates": float(X_train["ExitRates"].mean()),
        "mean_prod_duration": float(X_train["ProductRelated_Duration"].mean()),
        "hasil_knn": hasil_knn,
        "hasil_rf": hasil_rf,
    }


def encode_raw_input(raw_dict: dict) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLS}
    for col in SCALE_COLS:
        row[col] = raw_dict[col]
    row["Weekend"] = int(bool(raw_dict["Weekend"]))
    month_col = f"Month_{raw_dict['Month']}"
    if month_col in row:
        row[month_col] = 1
    visitor_col = f"VisitorType_{raw_dict['VisitorType']}"
    if visitor_col in row:
        row[visitor_col] = 1
    return pd.DataFrame([row], columns=FEATURE_COLS)


def scale_features(df_encoded: pd.DataFrame, scaler) -> pd.DataFrame:
    df_scaled = df_encoded.copy().astype(float)
    df_scaled[SCALE_COLS] = scaler.transform(df_encoded[SCALE_COLS])
    return df_scaled


def append_log(entries: list[dict]):
    os.makedirs(LOG_DIR, exist_ok=True)
    df_new = pd.DataFrame(entries)
    if os.path.exists(LOG_FILE):
        df_new.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_FILE, index=False)


def load_logs() -> pd.DataFrame:
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=[
        "timestamp", "source", "model", "threshold", "page_values",
        "exit_rates", "product_duration", "month", "visitor_type",
        "prob_beli", "prediksi",
    ])


# Load model & statistik referensi
rf_model, knn_model, scaler = load_artifacts()
ref = load_reference_stats()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("Project 12 ML")
    st.caption("Prediksi Perilaku Pembelian Pengunjung E-Commerce")
    st.divider()

    model_choice = st.selectbox(
        "Pilih Model Prediksi",
        ["Random Forest (100 trees) - Rekomendasi", "KNN (K=5)"],
    )
    threshold = st.slider(
        "Ambang Batas Probabilitas (Threshold)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Default 0.50. Turunkan threshold (mis. 0.35) jika perusahaan ingin meningkatkan Recall untuk menjaring lebih banyak calon pembeli.",
    )
    st.divider()
    st.markdown("**Ringkasan Dataset**")
    st.markdown(
        "- Total Sesi: `12.330` (28 fitur)\n"
        "- Split: `80:20 Stratified` (`random_state=42`)\n"
        f"- Baseline Pembeli Train: `{ref['buy_rate_train']*100:.2f}%`\n"
        "- Target Positif: `Revenue = 1` (Membeli)"
    )

# ==================== HEADER ====================
st.title("Dashboard Prediksi Niat Beli Pengunjung E-Commerce")
st.markdown(
    "Implementasi **Machine Learning Workflow (Tahap 6: Deployment & Tahap 7: Monitoring)** "
    "menggunakan dataset *UCI Online Shoppers Purchasing Intention*."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediksi Pengunjung",
    "Prediksi Batch (CSV)",
    "Evaluasi & Perbandingan Model",
    "Monitoring Model",
])

# ==================== TAB 1: LIVE INFERENCE ====================
with tab1:
    st.subheader("Simulasi Prediksi Sesi Pengunjung")
    st.write("Pilih preset contoh untuk demo cepat atau ubah langsung parameter aktivitas pengunjung di bawah ini.")

    if "form_vals" not in st.session_state:
        st.session_state.form_vals = PRESETS["Calon Pembeli Potensial"].copy()

    preset_cols = st.columns(len(PRESETS))
    for idx, (preset_name, preset_data) in enumerate(PRESETS.items()):
        if preset_cols[idx].button(preset_name, width="stretch"):
            st.session_state.form_vals = preset_data.copy()
            st.rerun()

    vals = st.session_state.form_vals

    with st.form("single_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**1. Aktivitas Halaman**")
            product_related = st.number_input("ProductRelated (Jumlah Halaman Produk)", 0, 1000, int(vals["ProductRelated"]))
            product_duration = st.number_input("ProductRelated_Duration (Detik)", 0.0, 70000.0, float(vals["ProductRelated_Duration"]), step=10.0)
            administrative = st.number_input("Administrative (Halaman Akun)", 0, 50, int(vals["Administrative"]))
            admin_duration = st.number_input("Administrative_Duration (Detik)", 0.0, 5000.0, float(vals["Administrative_Duration"]), step=5.0)
            informational = st.number_input("Informational (Halaman Info)", 0, 50, int(vals["Informational"]))
            info_duration = st.number_input("Informational_Duration (Detik)", 0.0, 5000.0, float(vals["Informational_Duration"]), step=5.0)

        with c2:
            st.markdown("**2. Metrik Google Analytics**")
            page_values = st.number_input(
                "PageValues (Fitur Terpenting #1)",
                0.0, 400.0, float(vals["PageValues"]), step=1.0,
                help="Rata-rata nilai ekonomi halaman yang dikunjungi sebelum transaksi.",
            )
            exit_rates = st.slider("ExitRates (Fitur Terpenting #2)", 0.0, 0.20, float(vals["ExitRates"]), step=0.001, format="%.3f")
            bounce_rates = st.slider("BounceRates", 0.0, 0.20, float(vals["BounceRates"]), step=0.001, format="%.3f")
            special_day = st.slider("SpecialDay (Kedekatan Hari Spesial)", 0.0, 1.0, float(vals["SpecialDay"]), step=0.2)

        with c3:
            st.markdown("**3. Konteks & Tipe Pengunjung**")
            month = st.selectbox("Month (Bulan Kunjungan)", MONTH_OPTIONS, index=MONTH_OPTIONS.index(vals["Month"]))
            visitor_type = st.selectbox("VisitorType", VISITOR_OPTIONS, index=VISITOR_OPTIONS.index(vals["VisitorType"]))
            weekend = st.checkbox("Kunjungan Akhir Pekan (Weekend)", value=bool(vals["Weekend"]))

            with st.expander("Parameter Teknis (Opsional)"):
                os_code = st.number_input("OperatingSystems (1-8)", 1, 8, int(vals["OperatingSystems"]))
                browser = st.number_input("Browser (1-13)", 1, 13, int(vals["Browser"]))
                region = st.number_input("Region (1-9)", 1, 9, int(vals["Region"]))
                traffic_type = st.number_input("TrafficType (1-20)", 1, 20, int(vals["TrafficType"]))

        submitted = st.form_submit_button("Prediksi Sekarang", type="primary", width="stretch")

    if submitted:
        raw_input = {
            "Administrative": administrative,
            "Administrative_Duration": admin_duration,
            "Informational": informational,
            "Informational_Duration": info_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "OperatingSystems": os_code,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "Month": month,
            "VisitorType": visitor_type,
            "Weekend": weekend,
        }
        st.session_state.form_vals = raw_input.copy()

        df_enc = encode_raw_input(raw_input)
        df_scl = scale_features(df_enc, scaler)

        prob_rf = float(rf_model.predict_proba(df_enc)[0][1])
        prob_knn = float(knn_model.predict_proba(df_scl)[0][1])

        is_rf = model_choice.startswith("Random Forest")
        active_prob = prob_rf if is_rf else prob_knn
        active_name = "Random Forest (100 trees)" if is_rf else "KNN (K=5)"
        pred_label = int(active_prob >= threshold)

        # Simpan ke log monitoring
        append_log([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "single",
            "model": active_name,
            "threshold": threshold,
            "page_values": page_values,
            "exit_rates": exit_rates,
            "product_duration": product_duration,
            "month": month,
            "visitor_type": visitor_type,
            "prob_beli": round(active_prob, 4),
            "prediksi": pred_label,
        }])

        st.divider()
        res_col1, res_col2 = st.columns([2, 1])
        with res_col1:
            if pred_label == 1:
                st.success(f"### ✅ HASIL PREDIKSI: AKAN MEMBELI (Revenue = 1)")
                st.write(
                    f"Model **{active_name}** memprediksi peluang pembelian sebesar **{active_prob*100:.2f}%** "
                    f"(di atas threshold `{threshold:.2f}`)."
                )
                st.info(
                    "💡 **Rekomendasi Aksi Bisnis:** Tampilkan *pop-up* voucher potongan harga, "
                    "gratis ongkir terbatas, atau pengingat *checkout* untuk mempercepat konversi."
                )
            else:
                st.warning(f"### 🛍️ HASIL PREDIKSI: TIDAK MEMBELI (Revenue = 0)")
                st.write(
                    f"Model **{active_name}** memprediksi peluang pembelian sebesar **{active_prob*100:.2f}%** "
                    f"(di bawah threshold `{threshold:.2f}`)."
                )
                st.info(
                    "💡 **Rekomendasi Aksi Bisnis:** Tidak perlu memberikan diskon agresif. "
                    "Tampilkan rekomendasi produk relevan untuk meningkatkan interaksi halaman."
                )

        with res_col2:
            st.markdown("**Perbandingan Kedua Model**")
            st.metric("Probabilitas RF (100 trees)", f"{prob_rf*100:.2f}%", "Beli" if prob_rf >= threshold else "Tidak Beli")
            st.metric("Probabilitas KNN (K=5)", f"{prob_knn*100:.2f}%", "Beli" if prob_knn >= threshold else "Tidak Beli")

        st.caption("Hasil prediksi ini otomatis dicatat ke tab **Monitoring Model** (`logs/prediction_log.csv`).")

# ==================== TAB 2: BATCH PREDICTION ====================
with tab2:
    st.subheader("Prediksi Massal (Batch Inference)")
    st.write(
        "Unggah file CSV berformat sama dengan dataset mentah (`online_shoppers_intention.csv`) "
        "atau gunakan sampel data uji bawaan untuk demo."
    )

    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload file CSV pengunjung", type=["csv"])
    with col_demo:
        st.write("")
        st.write("")
        use_sample = st.button("⚡ Demo 50 Sampel Data Test", width="stretch")

    df_batch_enc = None
    df_batch_display = None

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        if set(FEATURE_COLS).issubset(df_raw.columns):
            df_batch_enc = df_raw[FEATURE_COLS].copy()
        else:
            rows = []
            for _, r in df_raw.iterrows():
                rows.append(encode_raw_input(r.to_dict()).iloc[0])
            df_batch_enc = pd.DataFrame(rows, columns=FEATURE_COLS)
        df_batch_display = df_raw.copy()
    elif use_sample:
        X_test_sample = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv")).head(50)
        df_batch_enc = X_test_sample[FEATURE_COLS].copy()
        df_batch_display = X_test_sample[["ProductRelated", "ProductRelated_Duration", "ExitRates", "PageValues", "Weekend"]].copy()

    if df_batch_enc is not None:
        is_rf = model_choice.startswith("Random Forest")
        active_name = "Random Forest (100 trees)" if is_rf else "KNN (K=5)"
        if is_rf:
            probs = rf_model.predict_proba(df_batch_enc)[:, 1]
        else:
            probs = knn_model.predict_proba(scale_features(df_batch_enc, scaler))[:, 1]
        preds = (probs >= threshold).astype(int)

        df_out = df_batch_display.copy()
        df_out["Prob_Beli"] = np.round(probs, 4)
        df_out["Prediksi_Revenue"] = preds

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sesi Diprediksi", len(preds))
        m2.metric("Diprediksi Beli (1)", int(preds.sum()))
        m3.metric("Rasio Prediksi Beli", f"{preds.mean()*100:.2f}%")

        st.dataframe(df_out, width="stretch")

        # Catat ringkasan batch ke log monitoring
        log_entries = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for idx in range(len(df_batch_enc)):
            row_e = df_batch_enc.iloc[idx]
            month_val = next((m for m in MONTH_OPTIONS if row_e.get(f"Month_{m}", 0) == 1), "Unknown")
            vis_val = next((v for v in VISITOR_OPTIONS if row_e.get(f"VisitorType_{v}", 0) == 1), "Unknown")
            log_entries.append({
                "timestamp": ts,
                "source": "batch",
                "model": active_name,
                "threshold": threshold,
                "page_values": float(row_e["PageValues"]),
                "exit_rates": float(row_e["ExitRates"]),
                "product_duration": float(row_e["ProductRelated_Duration"]),
                "month": month_val,
                "visitor_type": vis_val,
                "prob_beli": round(float(probs[idx]), 4),
                "prediksi": int(preds[idx]),
            })
        append_log(log_entries)

        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            data=csv_bytes,
            file_name="hasil_prediksi_ecommerce.csv",
            mime="text/csv",
        )

# ==================== TAB 3: EVALUASI & PERBANDINGAN ====================
with tab3:
    st.subheader("Perbandingan Model: KNN (K=5) vs Random Forest (100 trees)")
    h_knn, h_rf = ref["hasil_knn"], ref["hasil_rf"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy (RF)", f"{h_rf['accuracy']*100:.2f}%", f"+{(h_rf['accuracy']-h_knn['accuracy'])*100:.2f}% vs KNN")
    k2.metric("Precision (RF)", f"{h_rf['precision']*100:.2f}%", f"+{(h_rf['precision']-h_knn['precision'])*100:.2f}% vs KNN")
    k3.metric("Recall (RF) ⭐", f"{h_rf['recall']*100:.2f}%", f"+{(h_rf['recall']-h_knn['recall'])*100:.2f}% vs KNN")
    k4.metric("F1-Score (RF)", f"{h_rf['f1']*100:.2f}%", f"+{(h_rf['f1']-h_knn['f1'])*100:.2f}% vs KNN")

    df_comp = pd.DataFrame([h_knn, h_rf]).set_index("model")
    st.dataframe(df_comp, width="stretch")

    st.info(
        "**Kesimpulan Bisnis:** Random Forest dipilih sebagai model utama karena unggul di seluruh metrik, "
        "terutama **Recall (53,93% vs 38,74%)** — mampu mendeteksi **206 dari 382 pembeli** (+58 calon pembeli lebih banyak dibanding KNN)."
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        p1 = os.path.join(OUTPUT_DIR, "13_perbandingan_metrik.png")
        if os.path.exists(p1):
            st.image(p1, caption="Perbandingan Metrik Evaluasi KNN vs Random Forest")
    with col_g2:
        p2 = os.path.join(OUTPUT_DIR, "11_feature_importance.png")
        if os.path.exists(p2):
            st.image(p2, caption="10 Fitur Terpenting pada Random Forest (PageValues paling dominan)")

    p3 = os.path.join(OUTPUT_DIR, "14_cm_perbandingan.png")
    if os.path.exists(p3):
        st.image(p3, caption="Confusion Matrix Berdampingan pada Data Test (n = 2.466)")

    with st.expander("Lihat Uji Sensitivitas Skala (KNN vs Random Forest)"):
        s1, s2 = st.columns(2)
        p_knn_s = os.path.join(OUTPUT_DIR, "09_knn_scaling.png")
        p_rf_s = os.path.join(OUTPUT_DIR, "12_rf_scaling.png")
        if os.path.exists(p_knn_s):
            s1.image(p_knn_s, caption="KNN sangat sensitif terhadap scaling")
        if os.path.exists(p_rf_s):
            s2.image(p_rf_s, caption="Random Forest tidak sensitif terhadap scaling")

# ==================== TAB 4: MONITORING MODEL ====================
with tab4:
    st.subheader("📈 Monitoring Performa & Deteksi Data Drift (Tahap 7 Workflow)")
    df_log = load_logs()

    if df_log.empty:
        st.info(
            "Belum ada log prediksi yang tercatat. Coba lakukan prediksi di tab **Prediksi Pengunjung** "
            "atau klik **Demo 50 Sampel Data Test** di tab **Prediksi Batch (CSV)** untuk melihat dashboard monitoring secara langsung."
        )
    else:
        total_req = len(df_log)
        pred_buy_rate = float(df_log["prediksi"].mean())
        avg_prob = float(df_log["prob_beli"].mean())
        avg_pv = float(df_log["page_values"].mean())

        buy_diff = pred_buy_rate - ref["buy_rate_train"]
        pv_diff = avg_pv - ref["mean_page_values"]

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Prediksi Tercatat", total_req)
        mc2.metric("Rasio Prediksi Beli (Live)", f"{pred_buy_rate*100:.2f}%", f"{buy_diff*100:+.2f}% vs Train ({ref['buy_rate_train']*100:.2f}%)")
        mc3.metric("Rata-rata Probabilitas Beli", f"{avg_prob*100:.2f}%")
        mc4.metric("Rata-rata PageValues (Live)", f"{avg_pv:.2f}", f"{pv_diff:+.2f} vs Train ({ref['mean_page_values']:.2f})")

        # Peringatan Drift Sederhana
        if abs(buy_diff) > 0.15:
            st.warning(
                f"⚠️ **Peringatan Prediction Drift:** Rasio prediksi beli saat ini ({pred_buy_rate*100:.2f}%) "
                f"berbeda lebih dari 15% dibanding baseline data latih ({ref['buy_rate_train']*100:.2f}%). "
                "Perlu evaluasi apakah terjadi perubahan pola belanja musiman atau pergeseran distribusi input."
            )
        else:
            st.success(
                "✅ **Status Distribusi Prediksi: Stabil.** Rasio prediksi masih berada dalam rentang wajar terhadap baseline data latih."
            )

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Distribusi Hasil Prediksi Tercatat**")
            counts = df_log["prediksi"].map({0: "Tidak Beli (0)", 1: "Beli (1)"}).value_counts()
            st.bar_chart(counts)
        with ch2:
            st.markdown("**Tren Probabilitas Prediksi (30 Request Terakhir)**")
            st.line_chart(df_log["prob_beli"].tail(30).reset_index(drop=True))

        st.markdown("**Riwayat Log Prediksi Terbaru**")
        st.dataframe(df_log.tail(20).iloc[::-1], width="stretch")

        act1, act2 = st.columns([1, 4])
        with act1:
            st.download_button(
                "⬇️ Download Log CSV",
                data=df_log.to_csv(index=False).encode("utf-8"),
                file_name="prediction_log.csv",
                mime="text/csv",
            )
        with act2:
            if st.button("🗑️ Reset Log Monitoring"):
                if os.path.exists(LOG_FILE):
                    os.remove(LOG_FILE)
                st.rerun()

    st.divider()
    st.markdown("### Strategi Monitoring & Pemeliharaan Model di Produksi")
    st.markdown(
        """
        1. **Logging Input & Prediksi:** Setiap permintaan prediksi dicatat ke `logs/prediction_log.csv` mencakup fitur kunci (`PageValues`, `ExitRates`, `ProductRelated_Duration`), probabilitas, dan hasil prediksi.
        2. **Pemantauan Data Drift:** Membandingkan rata-rata fitur utama (`PageValues` baseline = `{:.2f}`, `ExitRates` baseline = `{:.4f}`) pada data baru terhadap data latih secara berkala.
        3. **Pemantauan Concept Drift Musiman:** Karena konversi belanja sangat dipengaruhi bulan (misal November `25,35%` vs Februari `1,63%`), metrik **Recall** dan **Precision** aktual harus dievaluasi ulang setiap akhir bulan setelah label transaksi asli tersedia.
        4. **Kebijakan Retraining:** Jika **Recall** turun di bawah `45%` atau terjadi *prediction drift* signifikan secara konsisten, model dilatih ulang (*retraining*) dengan menambahkan data sesi transaksi terbaru.
        """.format(ref["mean_page_values"], ref["mean_exit_rates"])
    )
