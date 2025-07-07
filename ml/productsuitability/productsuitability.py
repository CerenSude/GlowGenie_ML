import os
os.environ["USE_SHAP_FORCE_CPU"] = "1"
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import shap
import pickle

# Loading CSV into DataFrame
df = pd.read_csv("backend/ml/productsuitability/products_dataset.csv", encoding="ISO-8859-9")

# Pivoting to wide format: one row per (ürün adı, kategori, cilt tipi), columns = each içerik
pivot_df = df.pivot_table(
    index=["ürün adı", "kategori", "cilt tipi"],
    columns="içerik",
    values="is_exist",
    fill_value=0
).reset_index()

# One-hotting skin types for convenience
skin_types = ["YAGLI", "KURU", "KARMA", "NORMAL"]
for skin in skin_types:
    pivot_df[skin] = pivot_df["cilt tipi"].apply(lambda x: 1 if x == skin else 0)

# Identifying ingredient columns (excluding identifiers and one-hot labels)
fixed_cols = ["ürün adı", "kategori", "cilt tipi"]
feature_cols = [c for c in pivot_df.columns if c not in fixed_cols + skin_types]

# Computing “pure” YAĞLI and KURU centroids (for optional pruning of YAĞLI)
pure_yagli_df = pivot_df[(pivot_df["cilt tipi"] == "YAGLI") & (~pivot_df["cilt tipi"].str.contains("-"))]
pure_kuru_df  = pivot_df[(pivot_df["cilt tipi"] == "KURU")  & (~pivot_df["cilt tipi"].str.contains("-"))]

yagli_centroid = pure_yagli_df[feature_cols].mean().values if len(pure_yagli_df) >= 1 else None
kuru_centroid  = pure_kuru_df[feature_cols].mean().values  if len(pure_kuru_df) >= 1  else None

# Defining a helper for SMOTE oversampling
def perform_smote(X: pd.DataFrame, y: pd.Series):
    sm = SMOTE(random_state=42)
    X_res_np, y_res = sm.fit_resample(X, y)
    X_res = pd.DataFrame(X_res_np, columns=X.columns)
    y_res = pd.Series(y_res)
    return X_res, y_res

# Training & storing each skin-type model + SHAP explainer + feature list
models_explainers = {}  # Will map skin_type -> (trained_model, shap_explainer, feature_list)

for skin in skin_types:
    print(f"\n📊 Training model for skin type: {skin}")

    # Filtering pure positives (cilt tipi == skin, no “-”) and pure negatives
    pos_df = pivot_df[(pivot_df["cilt tipi"] == skin) & (~pivot_df["cilt tipi"].str.contains("-"))].copy()
    neg_df = pivot_df[(~pivot_df["cilt tipi"].str.contains("-")) & (pivot_df["cilt tipi"] != skin)].copy()

    # Skipping if not enough examples remain
    if len(pos_df) < 10 or len(neg_df) < 10:
        print(f"⚠️ Skipping {skin}: not enough pure examples.")
        continue

    # Special pruning for YAĞLI only (clustering & removing cluster closest to KURU centroid)
    if skin == "YAGLI" and kuru_centroid is not None and len(pos_df) >= 10:
        Xp = pos_df[feature_cols].values
        kmeans = KMeans(n_clusters=2, random_state=42).fit(Xp)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        dists = np.linalg.norm(centers - kuru_centroid, axis=1)
        ambiguous_label = np.argmin(dists)
        pruned = pos_df.iloc[np.where(labels != ambiguous_label)[0]]
        # Only replacing pos_df if ≥50% remain and ≥10 rows
        if len(pruned) >= 0.5 * len(pos_df) and len(pruned) >= 10:
            pos_df = pruned

    # Re-checking counts after pruning
    if len(pos_df) < 10 or len(neg_df) < 10:
        print(f"⚠️ Skipping {skin}: not enough pure examples after pruning.")
        continue

    # Combining positives + negatives, building X and y
    data_df = pd.concat([pos_df, neg_df], axis=0)
    X = data_df[feature_cols].copy()
    y = pd.Series(np.where(data_df["cilt tipi"] == skin, 1, 0), index=data_df.index)

    # Stratified train/test splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE oversampling on the training fold
    X_res, y_res = perform_smote(X_train, y_train)

    # Training a RandomForest on the SMOTEd training set
    base_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    base_model.fit(X_res, y_res)

# Building SHAP explainer using the resampled training set
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_res)
# Extracting class-1 SHAP (some versions return list of arrays, others a 3D array)
if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_vals_class1 = shap_values[:, :, 1]
else:
    shap_vals_class1 = shap_values

# Feature selection via SHAP for YAĞLI & KARMA (top 15 by mean |SHAP|)
if skin in ["YAGLI", "KARMA"]:
    mean_abs_shap = pd.DataFrame({
        "feature": X_res.columns.tolist(),
        "mean_abs_shap": np.abs(shap_vals_class1).mean(axis=0)
    }).sort_values(by="mean_abs_shap", ascending=False)
    top15 = mean_abs_shap["feature"].head(15).tolist()
    print(f"📌 Top SHAP features for {skin}: {top15}")
    feature_list = top15
else:
    print(f"ℹ️ Skipping feature selection for {skin}")
    feature_list = X_res.columns.tolist()

# Retraining final RandomForest on the SMOTEd training set using only feature_list
X_res_sel = X_res[feature_list]
X_test_sel = X_test[feature_list]

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_res_sel, y_res)

# Storing (model, explainer, feature_list) for this skin type
models_explainers[skin] = (final_model, explainer, feature_list)

# Evaluating on hold-out test fold
y_pred = final_model.predict(X_test_sel)
y_proba = final_model.predict_proba(X_test_sel)[:, 1]

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Suitable", "Suitable"]))
try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"🔍 ROC-AUC Score: {auc:.3f}")
except ValueError:
    print("⚠️ ROC-AUC could not be computed.")

# Defining a function to explain a single product for a given skin
def explain_product(product_name: str, skin: str):
    if skin not in models_explainers:
        print(f"⚠️ No trained model for skin={skin}.")
        return None

    final_model, explainer, feature_list = models_explainers[skin]

    # Finding the pivoted row for this product
    row = pivot_df[pivot_df["ürün adı"] == product_name]
    if row.empty:
        print(f"⚠️ Product '{product_name}' not found in pivot_df.")
        return None
    row = row.iloc[0]

    # Building a single-row DataFrame of features in feature_list
    X_single = pd.DataFrame([row[feature_list].values], columns=feature_list)

    # Predicting probability
    prob = final_model.predict_proba(X_single)[0, 1]
    label = "Suitable" if prob >= 0.5 else "Not Suitable"

    # Computing SHAP values for this single example
    shap_vals = explainer.shap_values(X_single)
    if isinstance(shap_vals, list):
        shap_vals_class1 = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_vals_class1 = shap_vals[:, :, 1]
    else:
        shap_vals_class1 = shap_vals

    sv_series = pd.Series(shap_vals_class1[0], index=feature_list)
    sv_sorted = sv_series.abs().sort_values(ascending=False)

    print(f"\n❓ Explanation for '{product_name}' on model '{skin}':")
    print(f" • Predicted probability of Suitable = {prob:.3f}  →  Label: {label}")
    print(" • Top SHAP drivers (feature : SHAP value → direction):")
    for feat in sv_sorted.index[:5]:
        val = sv_series[feat]
        direction = "↑ Suitable" if val > 0 else "↓ Not Suitable"
        print(f"     - {feat:20s}: SHAP = {val:.3f} ({direction})")

    return prob, sv_series

# Defining a callback for when a new product is added
def on_new_product_added(product_name: str):
    
    #Triggered after a new product is inserted. Assumes pivot_df includes that product.
    
    for skin in skin_types:
        result = explain_product(product_name, skin)
        if result is None:
            continue
        prob, sv_series = result

        # Building JSON-like list of top 5 features with SHAP values
        top5 = sv_series.abs().sort_values(ascending=False).index[:5].tolist()
        top5_with_vals = [{"feature": f, "shap": float(sv_series[f])} for f in top5]

        print(f"\n→ Saved explanation for '{product_name}' as skin='{skin}':")
        print(f"   • probability = {prob:.3f}")
        print(f"   • top5 features & SHAP = {top5_with_vals}")

# Example: explain an existing product right now
explain_product("CeraVe Moisturizing Cream", "KURU")

# Saving models & explainers to disk under ./models/
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

for skin, (model, explainer, _) in models_explainers.items():
    model_path     = os.path.join(output_dir, f"model_{skin}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    explainer_path = os.path.join(output_dir, f"explainer_{skin}.pkl")
    with open(explainer_path, "wb") as f:
        pickle.dump(explainer, f)

print("Saved models and explainers under './models/'")
