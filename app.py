"""LAKSAMANA - Layanan Analisis Stok & Mitigasi Bencana Maritim Nasional
Pendekatan hibrida: KNN (klasifikasi kerentanan iklim maritim) + Linear Regression
(estimasi defisit dan kebutuhan buffer stock pangan)."""
import altair as alt, numpy as np, pandas as pd, streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FILE = "Dataset_Pangan_Iklim_Cleaned_Normalized.xlsx"
CLIM = ["curah_hujan_mm", "kecepatan_angin_maks_ms", "kecepatan_angin_rata_ms"]
LBL = {"curah_hujan_mm": "Curah hujan (mm)", "kecepatan_angin_maks_ms": "Angin maks (m/s)",
       "kecepatan_angin_rata_ms": "Angin rata-rata (m/s)", "kebutuhan": "Kebutuhan (ton/minggu)"}
KELAS = ["Rendah", "Sedang", "Tinggi"]
BATAS = (5, 30)                      # ton defisit/bulan: Rendah < 5 <= Sedang < 30 <= Tinggi
ZBUF = {"Rendah": 0.0, "Sedang": 1.0, "Tinggi": 1.645}   # pengali pengaman buffer per kelas
st.set_page_config(page_title="LAKSAMANA", page_icon="⚓", layout="wide")

@st.cache_data
def load(src):
    x = pd.ExcelFile(src)
    d = {s: x.parse(s) for s in ["2_Cleaning_Iklim", "4_Gabungan_Pangan_Iklim", "5_Parameter_Normalisasi",
                                 "6_Flag_Review", "8_Kontrol_Kualitas", "Kelayakan_Komoditas"]}
    d["4_Gabungan_Pangan_Iklim"]["tanggal_mulai"] = pd.to_datetime(d["4_Gabungan_Pangan_Iklim"]["tanggal_mulai"])
    return d

def kelas_dari_defisit(t):
    return KELAS[0] if t < BATAS[0] else KELAS[1] if t < BATAS[1] else KELAS[2]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
h2, h3 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: .02em; color: #0F2A43; }
.hero { background: linear-gradient(120deg, #0F2A43 0%, #14425e 60%, #1B8A9A 130%); color: #EEF4F6; padding: 26px 30px;
        border-radius: 6px; border-bottom: 4px solid #E8742C; margin-bottom: 18px; }
.hero h1 { font-family: 'Barlow Condensed', sans-serif; font-size: 54px; line-height: 1; margin: 0; letter-spacing: .04em; color: #EEF4F6; }
.hero p { margin: 6px 0 12px; opacity: .85; font-size: 16px; }
.chip { display: inline-block; font-size: 12px; font-weight: 600; padding: 3px 10px; margin-right: 6px; border: 1px solid #7fc4cf; color: #cfeaf0; border-radius: 2px; }
.chip.o { border-color: #E8742C; color: #ffc9a3; }
[data-testid="stMetric"] { background: #fff; border-left: 4px solid #1B8A9A; padding: 10px 14px; border-radius: 2px; box-shadow: 0 1px 2px rgba(15,42,67,.08); }
[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }
button[data-baseweb="tab"] { font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: #E8742C; }
div[data-baseweb="tab-highlight"] { background-color: #E8742C; }
section[data-testid="stSidebar"] { background: #DCE9ED; border-right: 1px solid #c3d6dc; }
[data-testid="stDataFrame"] { border: 1px solid #c3d6dc; border-radius: 2px; }
[data-testid="stAlert"] { border-radius: 2px; }
</style>
<div class="hero">
  <h1>⚓ LAKSAMANA</h1>
  <p>Layanan Analisis Stok &amp; Mitigasi Bencana Maritim Nasional | studi kasus Kota Tanjungpinang</p>
  <span class="chip">KNN</span><span class="chip">Linear Regression</span><span class="chip o">Peringatan Dini</span><span class="chip o">Buffer Stock</span>
</div>
""", unsafe_allow_html=True)
up = st.sidebar.file_uploader("Dataset (opsional)", type="xlsx")
try:
    D = load(up if up else FILE)
except Exception as e:
    st.error(f"Dataset '{FILE}' tidak ditemukan atau formatnya tidak sesuai: {e}"); st.stop()
iklim, gab, kel = D["2_Cleaning_Iklim"], D["4_Gabungan_Pangan_Iklim"], D["Kelayakan_Komoditas"]
layak = sorted(kel.loc[kel["Status kelayakan"] == "Layak", "Komoditas"])
df = gab[(gab["ada_data_iklim"] == True) & gab.komoditas.isin(layak)].dropna(subset=CLIM + ["surplus_hitung", "kebutuhan"]).copy()
df["defisit"] = (-df["surplus_hitung"]).clip(lower=0)
st.sidebar.write(f"Komoditas layak: **{len(layak)}** | baris model: **{len(df):,}**")

# ---- KNN: label kerentanan bulanan dari total defisit komoditas layak
bln = df.groupby("periode_iklim")["defisit"].sum().rename("defisit_total_ton").reset_index()
bln = bln.merge(iklim[["periode"] + CLIM], left_on="periode_iklim", right_on="periode").drop(columns="periode")
bln["kerentanan"] = bln["defisit_total_ton"].map(kelas_dari_defisit)
Xk, yk = bln[CLIM], bln["kerentanan"]

t1, te, t2, t3, t4, ti = st.tabs(["1. Data", "2. EDA", "3. KNN: Kerentanan Iklim", "4. Linear Regression: Defisit", "5. Peringatan & Buffer Stock", "6. Insight"])

with t1:
    st.subheader("Pengumpulan, pembersihan, dan normalisasi data")
    st.write("**Sumber:** data.go.id (stok pangan mingguan, 33 berkas) dan BMKG Stasiun Meteorologi Raja Haji Fisabilillah (iklim bulanan).")
    c = st.columns(4)
    c[0].metric("Baris data gabungan", f"{len(gab):,}"); c[1].metric("Komoditas", gab.komoditas.nunique())
    c[2].metric("Bulan iklim", len(iklim)); c[3].metric("Baris ditandai", len(D["6_Flag_Review"]))
    st.write("**Kontrol kualitas pembersihan**"); st.dataframe(D["8_Kontrol_Kualitas"], width="stretch", hide_index=True)
    a, b = st.columns(2)
    a.write("**Jenis masalah data (flag)**"); a.bar_chart(D["6_Flag_Review"]["jenis_masalah"].value_counts())
    b.write("**Kelayakan komoditas**"); b.dataframe(kel[["Komoditas", "% pasokan sama dgn minggu lalu", "Status kelayakan"]], hide_index=True)
    st.write("**Parameter normalisasi (Min-Max dan Z-Score)**"); st.dataframe(D["5_Parameter_Normalisasi"], width="stretch", hide_index=True)

with te:
    st.subheader("Data understanding dan Exploratory Data Analysis")
    st.write("**Catatan sumber data**")
    st.dataframe(pd.DataFrame({"Dataset": ["Ketersediaan pangan mingguan Kota Tanjungpinang", "Iklim bulanan Stasiun Meteorologi Raja Haji Fisabilillah (WMO 96091)"],
        "Sumber": ["data.go.id (Satu Data Indonesia)", "BMKG"], "Format": ["Excel (.xlsx), 33 berkas", "Excel (.xlsx), 1 berkas"],
        "Periode": ["2024–2026, mingguan", "Okt 2024–Sep 2026, bulanan"], "Baris": [len(gab), len(iklim)],
        "Variabel": [gab.shape[1], iklim.shape[1]], "URL": ["(isi URL halaman dataset)", "(isi URL halaman dataset)"]}), hide_index=True, width="stretch")
    miss = gab.isna().sum(); miss = miss[miss > 0].rename("jumlah kosong")
    c = st.columns(3); c[0].metric("Baris duplikat (tanggal + komoditas)", int(gab.duplicated(["tanggal_mulai", "komoditas"]).sum()))
    c[1].metric("Kolom dengan nilai kosong", len(miss)); c[2].metric("Total baris data gabungan", f"{len(gab):,}")
    if len(miss): st.dataframe(miss)
    st.write("**Statistik deskriptif (variabel numerik utama)**")
    st.dataframe(gab[[x for x in ["pasokan_total", "kebutuhan", "surplus_hitung", "rasio_pasokan_kebutuhan"] if x in gab]].describe().round(2), width="stretch")
    mm = df.groupby("periode_iklim").agg(rasio_median=("rasio_pasokan_kebutuhan", "median"), surplus_z=("surplus_hitung_zscore", "mean"),
        defisit_ton=("defisit", "sum")).reset_index().merge(iklim[["periode"] + CLIM], left_on="periode_iklim", right_on="periode").drop(columns="periode")
    a, b = st.columns(2)
    a.write("**Visualisasi 1: Median rasio pasokan/kebutuhan per komoditas** (dipotong di 12)")
    a.bar_chart(gab.groupby("komoditas").rasio_pasokan_kebutuhan.median().clip(upper=12).sort_values())
    nb = df[df.komoditas != "Beras"].groupby("periode_iklim")[["pasokan_total", "kebutuhan"]].sum()
    b.write("**Visualisasi 2: Pasokan vs kebutuhan bulanan** (komoditas layak tanpa Beras, ton)"); b.line_chart(nb)
    a, b = st.columns(2)
    a.write("**Visualisasi 3: Curah hujan bulanan (mm)**"); a.bar_chart(iklim.set_index("periode")["curah_hujan_mm"])
    b.write("**Visualisasi 4: Kecepatan angin bulanan (m/s)**"); b.line_chart(iklim.set_index("periode")[CLIM[1:]])
    a, b = st.columns(2)
    a.write("**Visualisasi 5: Distribusi surplus (Z-Score)**")
    h, e = np.histogram(df["surplus_hitung_zscore"].clip(-5, 5), bins=20); a.bar_chart(pd.Series(h, index=np.round(e[:-1], 1), name="jumlah"))
    b.write("**Visualisasi 6: Curah hujan vs rasio pangan bulanan**"); b.scatter_chart(mm, x="curah_hujan_mm", y="rasio_median")
    cr = mm[CLIM].apply(lambda c: mm[["rasio_median", "surplus_z", "defisit_ton"]].corrwith(c)).T.round(2)
    cr.index = [LBL[i] for i in cr.index]
    st.write("**Korelasi Pearson iklim vs pangan (n = 24 bulan)**")
    ch = cr.reset_index(names="iklim").melt("iklim", var_name="pangan", value_name="r")
    hm = alt.Chart(ch).mark_rect().encode(x="pangan:N", y="iklim:N", color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")))
    st.altair_chart(hm + hm.mark_text().encode(text=alt.Text("r:Q", format=".2f"), color=alt.value("black")), width="stretch")
    arah = lambda r: ("positif" if r > 0 else "negatif") + " " + ("sangat lemah" if abs(r) < .2 else "lemah" if abs(r) < .4 else "sedang" if abs(r) < .6 else "kuat")
    st.write("**Interpretasi:** " + "; ".join(f"{i} vs {c}: r = {cr.loc[i, c]:.2f} ({arah(cr.loc[i, c])})" for i in cr.index for c in cr.columns))
    st.info("Korelasi tidak otomatis berarti hubungan sebab-akibat, dan hanya dihitung dari 24 bulan data.")

with t2:
    st.subheader("KNN: klasifikasi pola kerentanan iklim maritim")
    st.write(f"Tiap bulan diberi kelas kerentanan berdasarkan total defisit pasokan komoditas layak: "
             f"**Rendah** (< {BATAS[0]} ton), **Sedang** ({BATAS[0]}–{BATAS[1]} ton), **Tinggi** (≥ {BATAS[1]} ton). "
             "KNN mempelajari pola iklim (hujan dan angin) yang berkaitan dengan tiap kelas. "
             "Karena hanya 24 bulan, evaluasi memakai Leave-One-Out Cross-Validation.")
    ks = [1, 3, 5, 7, 9]
    mdl = lambda k: make_pipeline(StandardScaler(), KNeighborsClassifier(k))
    acc = {k: accuracy_score(yk, cross_val_predict(mdl(k), Xk, yk, cv=LeaveOneOut())) for k in ks}
    k = st.select_slider("Nilai K (tetangga)", ks, 5)
    st.write("**Akurasi LOOCV per K**"); st.bar_chart(pd.Series(acc, name="akurasi"))
    pred = cross_val_predict(mdl(k), Xk, yk, cv=LeaveOneOut())
    maj = yk.value_counts(normalize=True).max()
    c = st.columns(2); c[0].metric("Akurasi KNN (LOOCV)", f"{accuracy_score(yk, pred):.2f}"); c[1].metric("Baseline kelas terbanyak", f"{maj:.2f}")
    cm = pd.DataFrame(confusion_matrix(yk, pred, labels=KELAS), index=[f"Aktual {x}" for x in KELAS], columns=[f"Prediksi {x}" for x in KELAS])
    st.write("**Confusion matrix**"); st.dataframe(cm)
    rep = pd.DataFrame(classification_report(yk, pred, labels=KELAS, output_dict=True, zero_division=0)).T.round(2)
    st.write("**Precision, Recall, F1-score**"); st.dataframe(rep)
    a_knn, maj_knn = accuracy_score(yk, pred), maj
    st.write("**Kelas per bulan**"); st.dataframe(bln.assign(prediksi_knn=pred).round(2), hide_index=True, width="stretch")
    st.info("Hubungan iklim dengan kerentanan di sini hanya pola statistik. Lonjakan defisit Des 2025–Mar 2026 juga bertepatan dengan kenaikan kebutuhan, jadi bukan bukti sebab-akibat.")

def fit_lr(kom, feats):
    d = df[df.komoditas == kom].sort_values("tanggal_mulai")
    return d, make_pipeline(StandardScaler(), LinearRegression())

with t3:
    st.subheader("Linear Regression: estimasi surplus/defisit (ton per minggu)")
    st.write("Target: `surplus_hitung` dalam ton. Fitur: curah hujan, angin maks, angin rata-rata, dan kebutuhan mingguan. "
             "Model dilatih per komoditas karena skala ton tiap komoditas sangat berbeda. Pasokan tidak dipakai agar tidak terjadi kebocoran data. "
             "Defisit = −surplus jika negatif.")
    FE = CLIM + ["kebutuhan"]
    kom = st.selectbox("Komoditas", layak, index=layak.index("Telur Ayam") if "Telur Ayam" in layak else 0)
    cut = st.slider("Persentase data latih (urut waktu)", 50, 90, 75)
    d, pipe = fit_lr(kom, FE); n = int(len(d) * cut / 100); tr, te = d.iloc[:n], d.iloc[n:]
    pipe.fit(tr[FE], tr.surplus_hitung); p = pipe.predict(te[FE]); y = te.surplus_hitung; base = np.full(len(y), tr.surplus_hitung.mean())
    r = pd.DataFrame({"Linear Regression": [mean_absolute_error(y, p), mean_squared_error(y, p) ** .5, r2_score(y, p)],
                      "Baseline (rata-rata latih)": [mean_absolute_error(y, base), mean_squared_error(y, base) ** .5, r2_score(y, base)]},
                     index=["MAE (ton)", "RMSE (ton)", "R²"]).round(3)
    st.write(f"Data latih: **{len(tr)}** minggu | Data uji: **{len(te)}** minggu"); st.dataframe(r)
    co = pd.Series(pipe[-1].coef_, index=[LBL[f] for f in FE]).round(3)
    a, b = st.columns(2)
    a.write("**Koefisien (fitur terstandar)**"); a.dataframe(co.rename("koefisien"))
    b.write("**Aktual vs prediksi (data uji)**"); b.scatter_chart(pd.DataFrame({"Aktual": y.values, "Prediksi": p}), x="Aktual", y="Prediksi")
    st.info("Jika R² rendah atau negatif, fitur yang tersedia belum cukup menjelaskan surplus komoditas ini. Itu temuan yang perlu dilaporkan apa adanya.")

with t4:
    st.subheader("Peringatan dini dan estimasi buffer stock")
    st.write("Alur hibrida: **KNN** menentukan tingkat kerentanan iklim, lalu **Linear Regression** mengestimasi defisit. "
             "Buffer stock = defisit prediksi + pengaman (pengali × RMSE model). Pengali: Rendah 0, Sedang 1, Tinggi 1,645 (kira-kira tingkat layanan 95%). "
             "Angka pengali adalah asumsi kebijakan, bukan hasil model.")
    a, b, c = st.columns(3)
    rain = a.number_input("Curah hujan bulanan (mm)", 0.0, 1000.0, 300.0)
    wmx = b.number_input("Angin maks (m/s)", 0.0, 30.0, 7.0); wav = c.number_input("Angin rata-rata (m/s)", 0.0, 15.0, 2.0)
    kom2 = st.selectbox("Komoditas", layak, key="k4", index=layak.index("Telur Ayam") if "Telur Ayam" in layak else 0)
    d2 = df[df.komoditas == kom2]
    keb = st.number_input("Kebutuhan mingguan (ton)", 0.0, 100000.0, float(round(d2.kebutuhan.median(), 1)))
    knn = mdl(5).fit(Xk, yk); xin = pd.DataFrame([[rain, wmx, wav]], columns=CLIM)
    kls = knn.predict(xin)[0]; pb = dict(zip(knn.classes_, knn.predict_proba(xin)[0]))
    lr = make_pipeline(StandardScaler(), LinearRegression()).fit(d2[FE], d2.surplus_hitung)
    rm = mean_squared_error(d2.surplus_hitung, lr.predict(d2[FE])) ** .5
    sp = float(lr.predict(pd.DataFrame([[rain, wmx, wav, keb]], columns=FE))[0]); dfs = max(0.0, -sp); buf = dfs + ZBUF[kls] * rm
    m = st.columns(4); m[0].metric("Kerentanan iklim (KNN)", kls); m[1].metric("Prediksi surplus (ton/minggu)", f"{sp:,.1f}")
    m[2].metric("Estimasi defisit (ton/minggu)", f"{dfs:,.1f}"); m[3].metric("Buffer stock disarankan (ton/minggu)", f"{buf:,.1f}")
    st.write("Probabilitas kelas KNN: " + ", ".join(f"{k} {v:.0%}" for k, v in pb.items()))
    if kls == "Tinggi" or dfs > 0:
        st.warning(f"**Siaga:** pertimbangkan menambah stok {kom2} sekitar {buf:,.1f} ton/minggu, percepat pengiriman sebelum cuaca memburuk, dan siapkan pemasok cadangan.")
    elif kls == "Sedang":
        st.warning(f"**Waspada:** pantau stok {kom2} mingguan dan siapkan cadangan {buf:,.1f} ton/minggu.")
    else:
        st.success(f"**Aman:** kondisi iklim berkerentanan rendah dan tidak ada defisit diprediksi untuk {kom2}.")
    st.caption(f"RMSE model (data latih penuh) untuk {kom2}: {rm:,.2f} ton. Hasil bersifat indikatif karena data iklim hanya 24 bulan.")

with ti:
    st.subheader("Insight, implikasi, dan rekomendasi")
    kurang = gab.groupby("komoditas").rasio_pasokan_kebutuhan.median(); kurang = kurang[kurang < 1].index.tolist()
    tertinggi = cr.abs().stack().idxmax()
    st.markdown(f"""
**Apa yang kita pelajari dari data**
- Komoditas dengan median pasokan di bawah kebutuhan: **{', '.join(kurang) if kurang else 'tidak ada'}**. Beras dan Jagung selalu surplus besar.
- Keterkaitan iklim–pangan terkuat di data: **{tertinggi[0]}** dengan **{tertinggi[1]}** (r = {cr.loc[tertinggi]:.2f}). Ini tergolong lemah hingga sedang.
- KNN: akurasi LOOCV **{a_knn:.2f}** dibanding baseline kelas terbanyak **{maj_knn:.2f}**. Pola iklim saja belum cukup membedakan tingkat kerentanan.
- Linear Regression: kinerja berbeda tiap komoditas (lihat tab 4), dan umumnya masih terbatas karena data iklim hanya 24 bulan.

**Implikasi**
- Pemantauan stok sebaiknya memprioritaskan komoditas yang sering defisit, bukan hanya mengandalkan cuaca.
- Lonjakan defisit Des 2025–Mar 2026 bertepatan dengan kenaikan kebutuhan, jadi faktor permintaan perlu dipantau bersama iklim.

**Rekomendasi**
1. Verifikasi manual data Bawang, Cabai, dan Ikan, karena pasokannya hampir selalu identik dengan minggu sebelumnya.
2. Tambah data kejadian cuaca ekstrem, gelombang, atau gangguan pelayaran agar analisis mendekati mitigasi bencana maritim.
3. Perpanjang periode data (lebih dari 24 bulan) sebelum model dipakai untuk keputusan stok.
4. Gunakan buffer stock di tab 5 sebagai indikasi awal, bukan angka final.

**Keterbatasan:** studi kasus Kota Tanjungpinang, korelasi bukan sebab-akibat, dan batas kelas kerentanan serta pengali buffer adalah asumsi.
""")
