"""
臺北市住宅推薦系統 — Streamlit App
資料：實價登錄 114Q1～115Q1
執行方式：
  1. 將本檔與五個 CSV 放在同一資料夾
  2. pip install streamlit pandas scikit-learn matplotlib
  3. streamlit run real_estate_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── 字型（支援中文）──────────────────────────────
import subprocess, os
def _load_zh_font():
    try:
        result = subprocess.run(["fc-list", ":lang=zh"],
                                capture_output=True, text=True)
        fonts = [l.split(":")[0] for l in result.stdout.strip().split("\n") if l]
        if fonts:
            fm.fontManager.addfont(fonts[0])
            prop = fm.FontProperties(fname=fonts[0])
            plt.rcParams["font.family"] = prop.get_name()
    except Exception:
        pass
_load_zh_font()

# ── 頁面設定 ─────────────────────────────────────
st.set_page_config(
    page_title="臺北市住宅推薦系統",
    page_icon="🏠",
    layout="wide",
)

# ── 資料載入與前處理（快取） ──────────────────────
CSV_FILES = [
    "114Q1_A_lvr_land_A.csv",
    "114Q2_A_lvr_land_A.csv",
    "114Q3_A_lvr_land_A.csv",
    "114Q4_A_lvr_land_A.csv",
    "115Q1_A_lvr_land_A.csv",
]
QUARTERS = ["114Q1", "114Q2", "114Q3", "114Q4", "115Q1"]

@st.cache_data(show_spinner="載入資料中…")
def load_data():
    script_dir = Path(__file__).parent
    dfs = []
    for fname, q in zip(CSV_FILES, QUARTERS):
        path = script_dir / fname
        if not path.exists():
            st.error(f"找不到檔案：{fname}，請確認 CSV 與本程式放在同一資料夾。")
            st.stop()
        df = pd.read_csv(path, skiprows=[1])
        df["季度"] = q
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)

    # 篩選住宅
    housing = raw[raw["交易標的"].isin(
        ["房地(土地+建物)", "房地(土地+建物)+車位"])].copy()

    # 數值轉換
    for col in ["總價元", "建物移轉總面積平方公尺",
                "車位總價元", "建築完成年月", "交易年月日"]:
        housing[col] = pd.to_numeric(housing[col], errors="coerce")
    housing["車位總價元"] = housing["車位總價元"].fillna(0)

    # 衍生特徵
    housing["總價萬元"] = (housing["總價元"] - housing["車位總價元"]) / 10000
    housing["坪數"]    = housing["建物移轉總面積平方公尺"] * 0.3025
    housing["單價萬元坪"] = housing["總價萬元"] / housing["坪數"]

    def calc_age(row):
        try:
            return (int(str(int(row["交易年月日"]))[:3])
                    - int(str(int(row["建築完成年月"]))[:3]))
        except Exception:
            return np.nan

    housing["屋齡"] = housing.apply(calc_age, axis=1)

    # 過濾異常值
    df = housing[
        (housing["總價萬元"]   > 50)  & (housing["總價萬元"]   < 30000) &
        (housing["坪數"]       > 3)   & (housing["坪數"]       < 500)   &
        (housing["單價萬元坪"] > 5)   & (housing["單價萬元坪"] < 400)   &
        (housing["屋齡"]       >= 0)  & (housing["屋齡"]       < 80)
    ].copy().reset_index(drop=True)

    return df

@st.cache_data(show_spinner="訓練 K-Means 模型…")
def train_kmeans(df: pd.DataFrame):
    features = ["總價萬元", "坪數", "單價萬元坪", "屋齡"]
    X = df[features].dropna()
    df_km = df.loc[X.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    df_km["cluster"] = kmeans.fit_predict(X_scaled)

    # 依均價排序賦予標籤
    price_by_cluster = df_km.groupby("cluster")["總價萬元"].mean().sort_values()
    label_map = {
        price_by_cluster.index[0]: "中古住宅型",
        price_by_cluster.index[1]: "都會首購型",
        price_by_cluster.index[2]: "高資產住宅型",
    }
    df_km["客群"] = df_km["cluster"].map(label_map)
    sil = silhouette_score(X_scaled, df_km["cluster"])

    return df_km, scaler, kmeans, sil

# ── 推薦函式 ──────────────────────────────────────
def recommend(df_km, scaler, budget, district, size, age, top_n=5):
    """依使用者條件推薦 Top-N 住宅"""
    # 行政區篩選（若選「不限」則不篩）
    if district != "不限":
        pool = df_km[df_km["鄉鎮市區"] == district].copy()
    else:
        pool = df_km.copy()

    if pool.empty:
        return pd.DataFrame()

    features = ["總價萬元", "坪數", "單價萬元坪", "屋齡"]
    pool = pool.dropna(subset=features).copy()

    # 使用者向量（單價填池中位數）
    user_unit = budget / size if size > 0 else pool["單價萬元坪"].median()
    user_vec = np.array([[budget, size, user_unit, age]])
    user_scaled = scaler.transform(user_vec)

    pool_scaled = scaler.transform(pool[features])
    dist = euclidean_distances(user_scaled, pool_scaled)[0]
    pool = pool.copy()
    pool["推薦分數"] = (1 / (1 + dist) * 100).round(1)
    pool["距離"] = dist.round(3)

    top = pool.nsmallest(top_n, "距離")[
        ["鄉鎮市區", "總價萬元", "坪數", "單價萬元坪", "屋齡", "客群", "推薦分數", "季度"]
    ].reset_index(drop=True)
    top.index += 1
    top.columns = ["行政區", "總價（萬）", "坪數", "單價（萬/坪）", "屋齡（年）", "客群", "推薦分數", "季度"]
    top["總價（萬）"] = top["總價（萬）"].round(0).astype(int)
    top["坪數"]      = top["坪數"].round(1)
    top["單價（萬/坪）"] = top["單價（萬/坪）"].round(1)
    top["屋齡（年）"] = top["屋齡（年）"].astype(int)
    return top

# ── 圖表函式 ──────────────────────────────────────
def plot_district_bar(df, col, title, color):
    stats = df.groupby("鄉鎮市區")[col].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(stats.index[::-1], stats.values[::-1], color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(col)
    ax.tick_params(labelsize=9)
    for i, v in enumerate(stats.values[::-1]):
        ax.text(v * 1.01, i, f"{v:.0f}", va="center", fontsize=8)
    plt.tight_layout()
    return fig

def plot_volume(df):
    stats = df.groupby("鄉鎮市區").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(stats.index[::-1], stats.values[::-1], color="#2E86AB")
    ax.set_title("各行政區成交量", fontsize=11)
    ax.set_xlabel("筆數")
    ax.tick_params(labelsize=9)
    for i, v in enumerate(stats.values[::-1]):
        ax.text(v + 1, i, str(v), va="center", fontsize=8)
    plt.tight_layout()
    return fig

def plot_kmeans_scatter(df_km):
    colors = {"都會首購型": "#2E86AB", "中古住宅型": "#F18F01", "高資產住宅型": "#A23B72"}
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for grp, sub in df_km.groupby("客群"):
        ax.scatter(sub["坪數"], sub["總價萬元"],
                   alpha=0.3, s=10, color=colors[grp], label=grp)
    ax.set_xlabel("坪數")
    ax.set_ylabel("總價（萬元）")
    ax.set_title("K-Means 分群結果", fontsize=11)
    ax.set_xlim(0, 250); ax.set_ylim(0, 20000)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

def plot_elbow(df):
    features = ["總價萬元", "坪數", "單價萬元坪", "屋齡"]
    X = df[features].dropna()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    ks = range(2, 7)
    inertias = []
    sils = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lb = km.fit_predict(X_s)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_s, lb))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    ax1.plot(list(ks), inertias, "o-", color="#2E86AB")
    ax1.axvline(3, color="#A23B72", linestyle="--", alpha=0.7, label="k=3")
    ax1.set_title("Elbow Method", fontsize=10)
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
    ax1.legend(fontsize=8)

    bars = ax2.bar(list(ks), sils,
                   color=["#F18F01" if k == 3 else "#2E86AB" for k in ks])
    for k, s in zip(ks, sils):
        ax2.text(k, s + 0.005, f"{s:.3f}", ha="center", fontsize=8)
    ax2.set_title("Silhouette Score", fontsize=10)
    ax2.set_xlabel("k"); ax2.set_ylim(0, 0.65)
    plt.tight_layout()
    return fig

def plot_quarterly(df):
    q_stats = df.groupby("季度").agg(
        成交量=("總價萬元", "count"),
        平均總價=("總價萬元", "mean")
    ).reindex(QUARTERS)
    fig, ax1 = plt.subplots(figsize=(5.5, 3.2))
    ax2 = ax1.twinx()
    ax1.bar(QUARTERS, q_stats["成交量"], color="#2E86AB", alpha=0.7, label="成交量")
    ax2.plot(QUARTERS, q_stats["平均總價"], "o-", color="#A23B72", linewidth=2, label="均總價（萬）")
    ax1.set_ylabel("成交量", color="#2E86AB", fontsize=9)
    ax2.set_ylabel("均總價（萬元）", color="#A23B72", fontsize=9)
    ax1.set_title("季度趨勢", fontsize=11)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════
df      = load_data()
df_km, scaler, kmeans, sil_score = train_kmeans(df)

# ── 側邊欄：使用者條件輸入 ────────────────────────
st.sidebar.title("🏠 條件輸入")
st.sidebar.caption("依條件推薦最相似的 Top-5 住宅")

districts = ["不限"] + sorted(df["鄉鎮市區"].unique().tolist())
sel_district = st.sidebar.selectbox("行政區", districts)

budget = st.sidebar.slider(
    "預算（萬元）",
    min_value=300, max_value=20000,
    value=3000, step=100
)
size = st.sidebar.slider(
    "坪數需求",
    min_value=5, max_value=200,
    value=35, step=1
)
age_max = st.sidebar.slider(
    "可接受屋齡上限（年）",
    min_value=0, max_value=15,
    value=10, step=1
)

run_btn = st.sidebar.button("🔍 開始推薦", use_container_width=True)

# ── 主畫面標題 ────────────────────────────────────
st.title("臺北市住宅推薦系統")
st.caption("資料來源：內政部實價登錄 114Q1～115Q1　　有效樣本：2,816筆　　模型：K-Means（k=3）")

tab1, tab2, tab3 = st.tabs(["📊 市場分析", "🤖 分群結果", "🏠 推薦結果"])

# ── Tab 1：市場分析 ────────────────────────────────
with tab1:
    st.subheader("臺北市住宅市場概況")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("有效樣本", f"{len(df):,} 筆")
    c2.metric("平均總價", f"{df['總價萬元'].mean():,.0f} 萬元")
    c3.metric("平均坪數", f"{df['坪數'].mean():.1f} 坪")
    c4.metric("平均單價", f"{df['單價萬元坪'].mean():.1f} 萬/坪")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_volume(df))
        st.pyplot(plot_district_bar(df, "單價萬元坪", "各行政區平均單價（萬元/坪）", "#F18F01"))
    with col2:
        st.pyplot(plot_district_bar(df, "總價萬元", "各行政區平均總價（萬元）", "#A23B72"))
        st.pyplot(plot_quarterly(df))

# ── Tab 2：分群結果 ────────────────────────────────
with tab2:
    st.subheader("K-Means 分群分析")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.pyplot(plot_elbow(df))
        st.caption(f"選定 k=3　Silhouette Score = **{sil_score:.3f}**")
    with col2:
        st.pyplot(plot_kmeans_scatter(df_km))

    st.divider()
    st.subheader("分群統計")

    summary = df_km.groupby("客群").agg(
        筆數=("總價萬元", "count"),
        平均總價萬元=("總價萬元", "mean"),
        平均坪數=("坪數", "mean"),
        平均單價萬元坪=("單價萬元坪", "mean"),
        平均屋齡=("屋齡", "mean"),
    ).round(1)
    summary["占比"] = (summary["筆數"] / summary["筆數"].sum() * 100).round(1).astype(str) + "%"
    summary["平均總價萬元"] = summary["平均總價萬元"].astype(int)
    st.dataframe(summary[["筆數", "占比", "平均總價萬元", "平均坪數", "平均單價萬元坪", "平均屋齡"]],
                 use_container_width=True)

    with st.expander("ℹ️ 為何不使用 Accuracy / F1 / Confusion Matrix？"):
        st.markdown("""
K-Means 是**非監督式分群**演算法，資料本身沒有預先標記的正確答案（Ground Truth），
因此無法計算以下監督式分類指標：

| 指標 | 原因不適用 |
|------|-----------|
| Accuracy | 需要已知正確標籤 |
| Precision / Recall / F1 | 需要已知正確標籤 |
| Confusion Matrix | 需要已知正確標籤 |

本研究改採分群專用指標：
- **Inertia（WCSS）**：群內平方和，越小代表群內凝聚度越高
- **Silhouette Score**：衡量群間分離程度，值域 −1 至 1，越接近 1 越好
        """)

# ── Tab 3：推薦結果 ────────────────────────────────
with tab3:
    st.subheader("住宅推薦結果")

    if run_btn or "rec_result" in st.session_state:
        if run_btn:
            result = recommend(df_km, scaler, budget, sel_district, size, age_max)
            st.session_state["rec_result"] = result
            st.session_state["rec_params"] = (budget, sel_district, size, age_max)
        else:
            result = st.session_state["rec_result"]

        params = st.session_state.get("rec_params", (budget, sel_district, size, age_max))
        st.caption(f"查詢條件：行政區 **{params[1]}**　預算 **{params[0]:,} 萬**　坪數 **{params[2]} 坪**　屋齡上限 **{params[3]} 年**")

        if result.empty:
            st.warning("所選行政區無符合條件的物件，請調整條件或選擇「不限」。")
        else:
            # 客群色彩標記
            def color_cluster(val):
                cmap = {"都會首購型": "background-color: #D6EAF8",
                        "中古住宅型": "background-color: #FEF9E7",
                        "高資產住宅型": "background-color: #FDEDEC"}
                return cmap.get(val, "")

            styled = result.style.applymap(color_cluster, subset=["客群"])
            st.dataframe(styled, use_container_width=True)

            # 推薦分析
            st.markdown("**推薦說明**")
            top1 = result.iloc[0]
            st.info(
                f"最相似物件位於 **{top1['行政區']}**，"
                f"總價 **{top1['總價（萬）']:,} 萬元**，"
                f"**{top1['坪數']} 坪**，"
                f"屋齡 **{top1['屋齡（年）']} 年**，"
                f"客群分類為「{top1['客群']}」，"
                f"推薦分數 **{top1['推薦分數']}**。"
            )
    else:
        st.info("請在左側輸入條件後，點擊「🔍 開始推薦」。")
        st.markdown("""
**系統說明：**
1. 在左側選擇行政區、輸入預算、坪數與可接受屋齡上限
2. 系統以標準化歐氏距離計算使用者條件與全部成交物件的相似度
3. 輸出推薦分數最高的 Top-5 住宅物件
        """)
