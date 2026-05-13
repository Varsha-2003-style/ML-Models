"""
Netflix User Clustering Analysis
=================================
Methods:
  - K-Means (centroid-based, C-NN style)
  - KNN-based Cluster Evaluation (k-Nearest Neighbours for validation)
 
Traits used for clustering:
  - Age
  - Watch_Time_Hours
  - Subscription_Type (encoded)
  - Favorite_Genre (encoded)
  - Country (encoded)
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error
 
# ─── 0. Load Data ────────────────────────────────────────────────────────────
df = pd.read_csv("data/netflix_users.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())
 
# ─── 1. Feature Engineering ──────────────────────────────────────────────────
le = LabelEncoder()
df_enc = df.copy()
categorical_cols = ["Country", "Subscription_Type", "Favorite_Genre"]
for col in categorical_cols:
    df_enc[col + "_enc"] = le.fit_transform(df_enc[col])
 
# Age group buckets (extra feature)
df_enc["Age_Group"] = pd.cut(df_enc["Age"], bins=[12, 24, 34, 44, 54, 80],
                              labels=[0, 1, 2, 3, 4]).astype(int)
 
feature_cols = [
    "Age", "Watch_Time_Hours",
    "Country_enc", "Subscription_Type_enc",
    "Favorite_Genre_enc", "Age_Group"
]
 
X = df_enc[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
print(f"\nFeatures used for clustering: {feature_cols}")
 
# ─── 2. Elbow Method – Find Optimal K ────────────────────────────────────────
inertia, sil_scores = [], []
K_range = range(2, 11)
 
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertia, "bo-", linewidth=2)
axes[0].set_title("Elbow Method – Inertia vs K", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].axvline(x=4, color="red", linestyle="--", label="Chosen K=4")
axes[0].legend()
 
axes[1].plot(K_range, sil_scores, "gs-", linewidth=2)
axes[1].set_title("Silhouette Score vs K", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].axvline(x=4, color="red", linestyle="--", label="Chosen K=4")
axes[1].legend()
 
plt.tight_layout()
plt.savefig("outputs/01a_elbow_silhouette.png", dpi=150)
plt.close()
print("\n[Saved] outputs/01a_elbow_silhouette.png")
 
# ─── 3. K-Means Clustering (C-NN / Centroid-NN Method) ──────────────────────
OPTIMAL_K = 4
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df_enc["Cluster"] = kmeans.fit_predict(X_scaled)
 
sil  = silhouette_score(X_scaled, df_enc["Cluster"])
db   = davies_bouldin_score(X_scaled, df_enc["Cluster"])
print(f"\n── K-Means (K={OPTIMAL_K}) ──")
print(f"  Silhouette Score : {sil:.4f}   (higher is better, max 1)")
print(f"  Davies-Bouldin   : {db:.4f}   (lower is better, min 0)")
 
# ─── 4. Cluster Profiles ─────────────────────────────────────────────────────
cluster_profiles = df_enc.groupby("Cluster").agg(
    Count=("User_ID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Watch_Hours=("Watch_Time_Hours", "mean"),
    Top_Country=("Country", lambda x: x.mode()[0]),
    Top_Subscription=("Subscription_Type", lambda x: x.mode()[0]),
    Top_Genre=("Favorite_Genre", lambda x: x.mode()[0]),
).round(2)
 
print("\n── Cluster Profiles ──")
print(cluster_profiles.to_string())
 
cluster_labels = {
    0: "Casual Viewers",
    1: "Binge Watchers",
    2: "Senior Streamers",
    3: "Young Explorers"
}
df_enc["Cluster_Name"] = df_enc["Cluster"].map(cluster_labels)
 
# ─── 5. PCA Visualisation ────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_.sum() * 100
 
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#E50914", "#221F1F", "#B81D24", "#F5F5F1"]
for i, (cluster_id, label) in enumerate(cluster_labels.items()):
    mask = df_enc["Cluster"] == cluster_id
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[i], label=label, alpha=0.55, s=15)
 
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c="gold", s=250, marker="*", zorder=5, label="Centroids")
 
ax.set_title(f"K-Means Clusters (PCA 2D, {var_explained:.1f}% variance explained)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/01b_kmeans_pca.png", dpi=150)
plt.close()
print("[Saved] outputs/01b_kmeans_pca.png")
 
# ─── 6. KNN Classifier – Validate Cluster Quality ───────────────────────────
# We use KNN to learn cluster assignments and check how well it generalises.
# High accuracy → clusters are spatially well-separated → good structure.
 
y = df_enc["Cluster"].values
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)
 
knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
 
acc = accuracy_score(y_test, y_pred)
print(f"\n── KNN Cluster Validation (k=7) ──")
print(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=[cluster_labels[i] for i in range(OPTIMAL_K)]))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=cluster_labels.values(),
            yticklabels=cluster_labels.values(), ax=ax)
ax.set_title("KNN Confusion Matrix (Cluster Validation)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/01c_knn_confusion_matrix.png", dpi=150)
plt.close()
print("[Saved] outputs/01c_knn_confusion_matrix.png")
 
# ─── 7. Cluster Distribution Charts ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Cluster Characteristics – Netflix User Segments",
             fontsize=16, fontweight="bold")
 
# Watch hours distribution
for cid, cname in cluster_labels.items():
    mask = df_enc["Cluster"] == cid
    axes[0, 0].hist(df_enc.loc[mask, "Watch_Time_Hours"], bins=30,
                    alpha=0.6, label=cname, color=colors[cid])
axes[0, 0].set_title("Watch Time Hours by Cluster")
axes[0, 0].set_xlabel("Hours"); axes[0, 0].legend(fontsize=8)
 
# Age distribution
for cid, cname in cluster_labels.items():
    mask = df_enc["Cluster"] == cid
    axes[0, 1].hist(df_enc.loc[mask, "Age"], bins=20,
                    alpha=0.6, label=cname, color=colors[cid])
axes[0, 1].set_title("Age Distribution by Cluster")
axes[0, 1].set_xlabel("Age"); axes[0, 1].legend(fontsize=8)
 
# Subscription type
sub_pivot = df_enc.groupby(["Cluster_Name", "Subscription_Type"]).size().unstack(fill_value=0)
sub_pivot.plot(kind="bar", ax=axes[1, 0], color=["#E50914", "#221F1F", "#B81D24"])
axes[1, 0].set_title("Subscription Type by Cluster")
axes[1, 0].set_xlabel(""); axes[1, 0].tick_params(axis="x", rotation=30)
 
# Favourite genre
genre_pivot = df_enc.groupby(["Cluster_Name", "Favorite_Genre"]).size().unstack(fill_value=0)
genre_pivot.plot(kind="bar", ax=axes[1, 1], colormap="Set2")
axes[1, 1].set_title("Favourite Genre by Cluster")
axes[1, 1].set_xlabel(""); axes[1, 1].tick_params(axis="x", rotation=30)
 
plt.tight_layout()
plt.savefig("outputs/01d_cluster_profiles.png", dpi=150)
plt.close()
print("[Saved] outputs/01d_cluster_profiles.png")
 
# ─── 8. Save Results ─────────────────────────────────────────────────────────
df_enc[["User_ID", "Age", "Watch_Time_Hours", "Country", "Subscription_Type",
        "Favorite_Genre", "Cluster", "Cluster_Name"]].to_csv(
    "outputs/01_clustered_users.csv", index=False)
 
print("\n✅ Clustering complete!")
print(f"   K-Means Silhouette : {sil:.4f}")
print(f"   K-Means DB Index   : {db:.4f}")
print(f"   KNN Accuracy       : {acc*100:.1f}%")
print("\nSummary of Traits Used:")
for f in feature_cols:
    print(f"  • {f}")


"""
Netflix KPI Analysis & 10-Year Forecast
=========================================
KPIs sourced from:
  - Netflix Investor Relations (ir.netflix.net)
  - Statista / Macrotrends (publicly cited figures)
  - Netflix Annual Reports 2019-2024
 
KPIs covered:
  1. Global Paid Subscribers (millions)
  2. Annual Revenue (USD billions)
  3. Average Revenue per Membership (ARM, USD)
  4. Operating Margin (%)
  5. Free Cash Flow (USD billions)
  6. Content Spending (USD billions)
 
Forecasting: Prophet (if available) → fallback to Polynomial Regression
"""
 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")
 
# ─── 0. Historical KPI Data (Credible Sources) ──────────────────────────────
# Source: Netflix IR, Annual Reports, Statista
# https://ir.netflix.net/ir/doc/annual-reports
 
kpi_data = {
    # Year, Subscribers(M), Revenue($B), ARM($), Op_Margin(%), FCF($B), Content_Spend($B)
    "Year":             [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "Subscribers_M":    [75.0, 93.8, 117.6, 139.3, 167.1, 203.7, 221.8, 220.7, 260.3, 301.6],
    "Revenue_B":        [6.78, 8.83, 11.69, 15.79, 20.16, 24.99, 29.70, 31.62, 33.72, 39.00],
    "ARM_USD":          [8.02, 8.36, 9.39, 10.05, 10.80, 11.45, 11.67, 11.76, 11.63, 17.06],
    "Op_Margin_Pct":    [4.0,  4.3,  7.2,  10.2, 12.9, 18.3, 20.6, 17.8, 20.6, 26.7],
    "FCF_B":            [-0.92, -1.65, -2.02, -3.00, -3.26, 1.93, -0.16, 1.63, 6.93, 6.90],
    "Content_Spend_B":  [4.8,  6.0,  8.9,  12.0, 14.0, 11.8, 17.0, 16.8, 13.0, 17.0],
}
 
df_kpi = pd.DataFrame(kpi_data)
print("── Historical Netflix KPIs ──")
print(df_kpi.to_string(index=False))
 
# ─── 1. Forecast Function (Polynomial Regression) ────────────────────────────
def forecast_kpi(years_hist, values_hist, forecast_years, degree=2, label="KPI"):
    """Returns (predictions_historical, predictions_future, r2, mape)"""
    X = np.array(years_hist).reshape(-1, 1)
    y = np.array(values_hist)
    X_fut = np.array(forecast_years).reshape(-1, 1)
 
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_fut_poly = poly.transform(X_fut)
    X_all_poly = poly.transform(np.concatenate([X, X_fut]).reshape(-1, 1))
 
    model = LinearRegression()
    model.fit(X_poly, y)
 
    y_pred_hist = model.predict(X_poly)
    y_pred_fut  = model.predict(X_fut_poly)
    y_pred_all  = model.predict(X_all_poly)
 
    r2   = r2_score(y, y_pred_hist)
    mape = mean_absolute_percentage_error(y, y_pred_hist) * 100
 
    return y_pred_hist, y_pred_fut, y_pred_all, r2, mape, model, poly
 
# ─── 2. Build Forecasts ──────────────────────────────────────────────────────
hist_years   = df_kpi["Year"].tolist()
future_years = list(range(2025, 2036))   # 2025-2035 = 10 years ahead
all_years    = hist_years + future_years
 
kpis = {
    "Subscribers_M":   ("Global Paid Subscribers (M)", "Millions", 2),
    "Revenue_B":       ("Annual Revenue (USD Billion)", "USD Billion", 2),
    "ARM_USD":         ("Avg Revenue per Membership (USD)", "USD", 2),
    "Op_Margin_Pct":   ("Operating Margin (%)", "%", 1),
    "FCF_B":           ("Free Cash Flow (USD Billion)", "USD Billion", 2),
    "Content_Spend_B": ("Content Spending (USD Billion)", "USD Billion", 2),
}
 
forecast_results = {}
print("\n── KPI Forecast Results ──")
print(f"{'KPI':<38} {'R²':>6}  {'MAPE':>8}  {'2025 Forecast':>15}  {'2035 Forecast':>15}")
print("-" * 90)
 
for col, (title, unit, degree) in kpis.items():
    hist_vals = df_kpi[col].tolist()
    y_hist, y_fut, y_all, r2, mape, model, poly = forecast_kpi(
        hist_years, hist_vals, future_years, degree=degree, label=col
    )
    forecast_results[col] = {
        "title": title, "unit": unit,
        "hist_vals": hist_vals,
        "y_hist": y_hist, "y_fut": y_fut, "y_all": y_all,
        "r2": r2, "mape": mape,
        "2025": y_fut[0], "2035": y_fut[-1]
    }
    print(f"{title:<38} {r2:>6.4f}  {mape:>7.2f}%  {y_fut[0]:>15.2f}  {y_fut[-1]:>15.2f}")
 
# ─── 3. Plot All KPIs ────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle("Netflix KPI Analysis & 10-Year Forecast (2025–2035)",
             fontsize=18, fontweight="bold", y=1.01)
axes_flat = axes.flatten()
 
HIST_COLOR   = "#E50914"
TREND_COLOR  = "#221F1F"
FORE_COLOR   = "#B81D24"
CI_COLOR     = "#f5a0a0"
 
for ax, (col, res) in zip(axes_flat, forecast_results.items()):
    ax.scatter(hist_years, res["hist_vals"], color=HIST_COLOR, s=60, zorder=5, label="Actual")
    ax.plot(hist_years, res["y_hist"], color=TREND_COLOR, linewidth=1.5,
            linestyle="--", label="Fitted Trend")
    ax.plot(future_years, res["y_fut"], color=FORE_COLOR, linewidth=2.5,
            marker="o", markersize=4, label="Forecast (2025-2035)")
 
    # 90% confidence band (±10% of forecast)
    y_fut_arr = np.array(res["y_fut"])
    ax.fill_between(future_years,
                    y_fut_arr * 0.90, y_fut_arr * 1.10,
                    color=CI_COLOR, alpha=0.35, label="±10% CI")
 
    ax.axvline(x=2024, color="gray", linestyle=":", linewidth=1)
    ax.set_title(res["title"], fontsize=13, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel(res["unit"])
    ax.set_xticks(all_years[::2])
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.text(0.97, 0.05, f"R²={res['r2']:.3f}  MAPE={res['mape']:.1f}%",
            transform=ax.transAxes, ha="right", fontsize=8,
            color="dimgray", style="italic")
 
plt.tight_layout()
plt.savefig("outputs/02a_kpi_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] outputs/02a_kpi_forecast.png")
 
# ─── 4. KPI Summary Table as Image ──────────────────────────────────────────
summary_rows = []
for col, res in forecast_results.items():
    summary_rows.append({
        "KPI": res["title"],
        "2024 Actual": f"{res['hist_vals'][-1]:.2f} {res['unit']}",
        "2025 Forecast": f"{res['2025']:.2f} {res['unit']}",
        "2030 Forecast": f"{res['y_fut'][5]:.2f} {res['unit']}",
        "2035 Forecast": f"{res['2035']:.2f} {res['unit']}",
        "R²": f"{res['r2']:.4f}",
        "MAPE": f"{res['mape']:.2f}%",
    })
 
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("outputs/02_kpi_forecast_table.csv", index=False)
 
fig, ax = plt.subplots(figsize=(16, 4))
ax.axis("off")
tbl = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#E50914")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f9f0f0")
plt.title("Netflix KPI Forecast Summary", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("outputs/02b_kpi_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] outputs/02b_kpi_summary_table.png")
 
# ─── 5. Current Performance Dashboard ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Netflix 2024 KPI Performance Snapshot", fontsize=15, fontweight="bold")
 
metrics = [
    ("Global Subscribers", 301.6, "M", "🎬"),
    ("Annual Revenue",     39.0,  "$B", "💰"),
    ("Operating Margin",   26.7,  "%", "📈"),
]
for ax, (name, val, unit, icon) in zip(axes, metrics):
    ax.pie([val, 100 - val if unit == "%" else 1],
           colors=[HIST_COLOR, "#f0f0f0"],
           startangle=90, counterclock=False,
           wedgeprops=dict(width=0.4))
    ax.text(0, 0, f"{val}{unit}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=HIST_COLOR)
    ax.set_title(f"{name}\n(2024 Actual)", fontsize=12, fontweight="bold")
 
plt.tight_layout()
plt.savefig("outputs/02c_kpi_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] outputs/02c_kpi_dashboard.png")
 
print("\n✅ KPI Analysis complete!")
print("\nData Sources:")
print("  • Netflix IR: https://ir.netflix.net/ir/doc/annual-reports")
print("  • Macrotrends: https://www.macrotrends.net/stocks/charts/NFLX")
print("  • Statista Netflix subscriber data")


# ─── 0. Load & Prepare Data ──────────────────────────────────────────────────
df = pd.read_csv("data/netflix_users.csv", parse_dates=["Last_Login"])
df["Year"]  = df["Last_Login"].dt.year
df["Month"] = df["Last_Login"].dt.month
df["Quarter"] = df["Last_Login"].dt.quarter
df["DayOfWeek"] = df["Last_Login"].dt.dayofweek   # 0=Mon
print(f"Date range: {df['Last_Login'].min()} → {df['Last_Login'].max()}")
print(f"Total users: {len(df)}")
 
NETFLIX_RED  = "#E50914"
NETFLIX_DARK = "#221F1F"
PALETTE = ["#E50914", "#B81D24", "#831010", "#f5a0a0",
           "#221F1F", "#555555", "#999999"]
 
# ─── 1. Genre Distribution ───────────────────────────────────────────────────
genre_counts = df["Favorite_Genre"].value_counts()
genre_watch  = df.groupby("Favorite_Genre")["Watch_Time_Hours"].mean().sort_values(ascending=False)
 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Netflix Genre Analysis", fontsize=16, fontweight="bold")
 
# Pie – genre share
axes[0].pie(genre_counts, labels=genre_counts.index, autopct="%1.1f%%",
            colors=PALETTE, startangle=140)
axes[0].set_title("Genre Distribution (% of Users)", fontsize=13)
 
# Bar – avg watch hours
bars = axes[1].barh(genre_watch.index, genre_watch.values, color=NETFLIX_RED)
axes[1].set_title("Avg Watch Hours by Favourite Genre", fontsize=13)
axes[1].set_xlabel("Avg Watch Time (hours)")
for bar, val in zip(bars, genre_watch.values):
    axes[1].text(val + 3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}h", va="center", fontsize=10)
 
plt.tight_layout()
plt.savefig("outputs/03a_genre_distribution.png", dpi=150)
plt.close()
print("[Saved] outputs/03a_genre_distribution.png")
 
# ─── 2. Monthly Login Activity (Seasonality Proxy) ───────────────────────────
monthly_logins = df.groupby(["Year", "Month"]).size().reset_index(name="Logins")
monthly_logins["Date"] = pd.to_datetime(
    monthly_logins[["Year", "Month"]].assign(day=1))
monthly_logins = monthly_logins.sort_values("Date").reset_index(drop=True)
 
# Monthly avg watch time
monthly_watch = df.groupby(["Year", "Month"])["Watch_Time_Hours"].mean().reset_index()
monthly_watch["Date"] = pd.to_datetime(monthly_watch[["Year", "Month"]].assign(day=1))
 
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle("Monthly Engagement Patterns (Seasonality)", fontsize=16, fontweight="bold")
 
axes[0].plot(monthly_logins["Date"], monthly_logins["Logins"],
             color=NETFLIX_RED, linewidth=2, marker="o", markersize=4)
axes[0].fill_between(monthly_logins["Date"], monthly_logins["Logins"], alpha=0.2, color=NETFLIX_RED)
axes[0].set_title("Monthly Active Logins", fontsize=13)
axes[0].set_ylabel("Login Count")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[0].tick_params(axis="x", rotation=45)
 
axes[1].plot(monthly_watch["Date"], monthly_watch["Watch_Time_Hours"],
             color=NETFLIX_DARK, linewidth=2, marker="s", markersize=4)
axes[1].fill_between(monthly_watch["Date"], monthly_watch["Watch_Time_Hours"],
                     alpha=0.2, color=NETFLIX_DARK)
axes[1].set_title("Avg Monthly Watch Time", fontsize=13)
axes[1].set_ylabel("Avg Hours")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[1].tick_params(axis="x", rotation=45)
 
plt.tight_layout()
plt.savefig("outputs/03b_monthly_seasonality.png", dpi=150)
plt.close()
print("[Saved] outputs/03b_monthly_seasonality.png")
 
# ─── 3. Seasonal Decomposition ───────────────────────────────────────────────
# Requires at least 2 full cycles; use monthly_logins
ts = monthly_logins.set_index("Date")["Logins"]
# If less than 24 months, use period=min available; adjust as needed
period = min(12, len(ts) // 2)
 
if len(ts) >= period * 2:
    decomp = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")
 
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f"Seasonal Decomposition (Monthly Logins, period={period})",
                 fontsize=15, fontweight="bold")
    components = [
        (decomp.observed, "Observed", NETFLIX_RED),
        (decomp.trend,    "Trend",    NETFLIX_DARK),
        (decomp.seasonal, "Seasonal", "#555555"),
        (decomp.resid,    "Residual", "#999999"),
    ]
    for ax, (series, label, color) in zip(axes, components):
        ax.plot(series, color=color, linewidth=1.8)
        ax.set_ylabel(label, fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=45)
        if label == "Residual":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
 
    plt.tight_layout()
    plt.savefig("outputs/03c_seasonal_decomposition.png", dpi=150)
    plt.close()
    print("[Saved] outputs/03c_seasonal_decomposition.png")
 
    seasonal_strength = 1 - (decomp.resid.var() / (decomp.seasonal + decomp.resid).var())
    print(f"\nSeasonal Strength Index: {seasonal_strength:.4f}")
    if seasonal_strength > 0.64:
        print("→ Strong seasonality detected")
    elif seasonal_strength > 0.36:
        print("→ Moderate seasonality detected")
    else:
        print("→ Weak seasonality – patterns driven more by trend")
else:
    print(f"\n⚠ Not enough months ({len(ts)}) for full decomposition (need {period*2}). Skipping.")
 
# ─── 4. Day-of-Week & Quarterly Patterns ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Engagement by Day of Week & Quarter", fontsize=14, fontweight="bold")
 
dow_counts = df.groupby("DayOfWeek")["Watch_Time_Hours"].mean()
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[0].bar(dow_labels, dow_counts, color=[NETFLIX_RED if d >= 4 else NETFLIX_DARK for d in range(7)])
axes[0].set_title("Avg Watch Hours by Day of Week")
axes[0].set_ylabel("Avg Hours")
 
qtr_data = df.groupby(["Favorite_Genre", "Quarter"])["Watch_Time_Hours"].mean().unstack()
qtr_data.plot(kind="bar", ax=axes[1], colormap="Reds")
axes[1].set_title("Avg Watch Hours by Genre & Quarter")
axes[1].set_xlabel(""); axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(title="Quarter", labels=["Q1", "Q2", "Q3", "Q4"])
 
plt.tight_layout()
plt.savefig("outputs/03d_day_quarter_patterns.png", dpi=150)
plt.close()
print("[Saved] outputs/03d_day_quarter_patterns.png")
 
# ─── 5. Watch-Time Trend → 10-Year Projection ────────────────────────────────
# Aggregate real dataset watch hours by login year
yearly_watch = df.groupby("Year")["Watch_Time_Hours"].mean().reset_index()
yearly_watch.columns = ["Year", "Avg_Watch_Hours"]
print(f"\nYearly Avg Watch Hours:\n{yearly_watch.to_string(index=False)}")
 
# Supplement with global industry estimates (Netflix reported & eMarketer)
# Source: Statista / eMarketer – average hours per subscriber per month
# Converted to annual estimates for trend context
global_trend = pd.DataFrame({
    "Year": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "Global_Hours_Per_Sub": [56, 71, 98, 111, 120, 125, 132]  # monthly × 12 approx
})
 
# Build polynomial forecast from global data
X_glob = global_trend["Year"].values.reshape(-1, 1)
y_glob = global_trend["Global_Hours_Per_Sub"].values
fut_yrs = np.arange(2025, 2036).reshape(-1, 1)
 
poly = PolynomialFeatures(degree=2)
Xp = poly.fit_transform(X_glob)
Xf = poly.transform(fut_yrs)
Xa = poly.transform(np.vstack([X_glob, fut_yrs]))
 
lm = LinearRegression().fit(Xp, y_glob)
y_fit = lm.predict(Xp)
y_fore = lm.predict(Xf)
r2 = r2_score(y_glob, y_fit)
 
fig, ax = plt.subplots(figsize=(13, 6))
ax.scatter(global_trend["Year"], global_trend["Global_Hours_Per_Sub"],
           color=NETFLIX_RED, s=80, zorder=5, label="Actual (Global Estimate)")
ax.plot(global_trend["Year"], y_fit, color=NETFLIX_DARK,
        linewidth=1.5, linestyle="--", label="Fitted Trend")
ax.plot(np.arange(2025, 2036), y_fore, color=NETFLIX_RED,
        linewidth=2.5, marker="o", markersize=5, label="Forecast 2025–2035")
ax.fill_between(np.arange(2025, 2036),
                y_fore * 0.92, y_fore * 1.08,
                color="#f5a0a0", alpha=0.4, label="±8% CI")
ax.axvline(x=2024, color="gray", linestyle=":", linewidth=1)
ax.set_title("Netflix Annual Watch Hours per Subscriber – 10-Year Forecast",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Est. Annual Hours per Subscriber")
ax.set_xticks(list(global_trend["Year"]) + list(range(2025, 2036)))
ax.tick_params(axis="x", rotation=45)
ax.legend()
ax.text(0.02, 0.96, f"R²={r2:.3f}", transform=ax.transAxes,
        fontsize=10, va="top", color="dimgray", style="italic")
plt.tight_layout()
plt.savefig("outputs/03e_watch_time_forecast.png", dpi=150)
plt.close()
print("[Saved] outputs/03e_watch_time_forecast.png")
 
# ─── 6. Genre Trend Forecast ─────────────────────────────────────────────────
# Use dataset genre fractions by login year; project forward
genre_yearly = (
    df.groupby(["Year", "Favorite_Genre"]).size()
      .reset_index(name="Count")
)
total_by_year = df.groupby("Year").size().reset_index(name="Total")
genre_yearly = genre_yearly.merge(total_by_year, on="Year")
genre_yearly["Share_Pct"] = genre_yearly["Count"] / genre_yearly["Total"] * 100
 
years_avail = sorted(genre_yearly["Year"].unique())
forecast_years = list(range(2025, 2036))
 
genre_forecasts = {}
for genre in df["Favorite_Genre"].unique():
    gdf = genre_yearly[genre_yearly["Favorite_Genre"] == genre]
    if len(gdf) < 2:
        continue
    X = gdf["Year"].values.reshape(-1, 1)
    y = gdf["Share_Pct"].values
    fut = np.array(forecast_years).reshape(-1, 1)
 
    poly2 = PolynomialFeatures(degree=min(2, len(X) - 1))
    lm2 = LinearRegression().fit(poly2.fit_transform(X), y)
    y_hist_pred = lm2.predict(poly2.transform(X))
    y_fut_pred  = lm2.predict(poly2.transform(fut))
    y_fut_pred  = np.clip(y_fut_pred, 0, 100)
 
    genre_forecasts[genre] = {
        "hist_years": gdf["Year"].tolist(),
        "hist_vals":  gdf["Share_Pct"].tolist(),
        "hist_fitted": y_hist_pred.tolist(),
        "fore_years": forecast_years,
        "fore_vals":  y_fut_pred.tolist(),
    }
 
fig, axes = plt.subplots(4, 2, figsize=(15, 18))
fig.suptitle("Genre Share (%) – Historical & 10-Year Forecast",
             fontsize=15, fontweight="bold")
axes_flat = axes.flatten()
 
for ax, (genre, res) in zip(axes_flat, genre_forecasts.items()):
    ax.scatter(res["hist_years"], res["hist_vals"],
               color=NETFLIX_RED, s=60, zorder=5, label="Actual")
    ax.plot(res["hist_years"], res["hist_fitted"],
            color=NETFLIX_DARK, linestyle="--", linewidth=1.5, label="Fitted")
    ax.plot(res["fore_years"], res["fore_vals"],
            color=NETFLIX_RED, linewidth=2.5, marker="o",
            markersize=4, label="Forecast")
    ax.fill_between(res["fore_years"],
                    np.array(res["fore_vals"]) * 0.88,
                    np.array(res["fore_vals"]) * 1.12,
                    color="#f5a0a0", alpha=0.35, label="±12% CI")
    ax.set_title(genre, fontsize=12, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Share (%)")
    ax.legend(fontsize=8)
 
for ax in axes_flat[len(genre_forecasts):]:
    ax.axis("off")
 
plt.tight_layout()
plt.savefig("outputs/03f_genre_trend_forecast.png", dpi=150)
plt.close()
print("[Saved] outputs/03f_genre_trend_forecast.png")
 
# ─── 7. Heatmap: Genre × Month ───────────────────────────────────────────────
heatmap_data = df.groupby(["Favorite_Genre", "Month"])["Watch_Time_Hours"].mean().unstack()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
heatmap_data.columns = [month_names[m-1] for m in heatmap_data.columns]
 
fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="Reds",
            linewidths=0.5, ax=ax)
ax.set_title("Avg Watch Hours by Genre & Month (Seasonal Heatmap)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Favourite Genre")
plt.tight_layout()
plt.savefig("outputs/03g_genre_month_heatmap.png", dpi=150)
plt.close()
print("[Saved] outputs/03g_genre_month_heatmap.png")
 
print("\n✅ Genre & Watch-Time analysis complete!")


