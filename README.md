# GlowGenie_ML
Machine Learning component of the Glow Genie project.

 Skin Type Prediction Model

This machine learning model is designed to predict a user's skin type based on their answers to a short 9-question survey. It is part of the GlowGenie skincare recommendation system.
📊 What the Model Does

The model classifies users into one of four skin types:

    Oily

    Dry

    Combination

    Normal

It uses a logistic regression classifier trained on both real and synthetically generated data, allowing better balance and generalization across underrepresented classes.

🗃️ Dataset Information
📥 Source

The dataset was collected via a Google Forms survey, consisting of 160 responses to 9 multiple-choice questions related to the user's skin condition and experience.

🧪 Survey Questions:

The questions included items like:

    How often does your skin feel oily, especially in the T-zone?

    Does your skin feel dry after cleansing?

    How do moisturizers feel on your skin?
    ... (see full question list in SkinTypePrediction_Dataset.csv)

The 9th question asked users to label their skin type directly. These self-reported labels served as ground truth for supervised learning.

🧹 Data Preprocessing Steps

    Dropped irrelevant columns such as timestamp and email.

    Renamed columns for clarity and consistency.

    Standardized label values (dry (kuru) → dry, etc.).

    Moved the target label (SkinType) to the last column.

    Encoded categorical features using one-hot encoding.

    Encoded the label using LabelEncoder.

📈 Data Augmentation with CTGAN

To overcome class imbalance in the dataset, CTGAN (a GAN-based tabular data generator) was used to create synthetic samples for each class:
Skin Type	Real Samples	Synthetic Samples	Total
Dry	~10	130	~140
Normal	~23	117	~140
Oily	~28	112	~140
Combination	~87	53	~140

This produced a balanced dataset with roughly 140 samples per class.

🧠 Model Training

    Model Used: LogisticRegression (L2 regularization)

    Training/Test Split: 80/20 with stratified sampling

    Evaluation Metrics:

        Accuracy

        Classification report

        Confusion matrix

You can find sample output metrics in the terminal logs when running the script.

💾 Model Export

The trained model is saved as:

skintypeprediction.pkl

It can be loaded using joblib.load() in the main backend of the GlowGenie system for real-time predictions.

🛠️ Alternatives Considered (Commented in Code)

Other models were tested but not finalized, including:

    RandomForestClassifier

    XGBoostClassifier

    LogisticRegression with GridSearchCV for hyperparameter tuning

These experiments are included as commented code for future testing or performance comparison.

-----------------------------------------------------------------------------------------------

Product Suitability Prediction Model

This machine learning model determines whether a skincare product is suitable for a specific skin type (KURU, YAĞLI, KARMA, NORMAL). It is the second model of the GlowGenie system and works by analyzing the product’s ingredients.

🔍 What the Model Does

For each skin type, a separate Random Forest classifier is trained to predict whether a product is suitable (1) or not suitable (0) for that skin type. Suitability is determined per skin type, meaning each product is evaluated against each skin type independently.

Each model is:

    Balanced using SMOTE (Synthetic Minority Over-sampling Technique)

    Fine-tuned by selecting top SHAP features

    Saved with its own SHAP explainer for transparency

📊 Dataset Information
📥 Source

The product dataset was created manually by collecting publicly available skincare product data. Each row corresponds to a single product–skin type pair and contains the following fields:

    ürün adı: Product name

    kategori: Category (moisturizer, cleanser, etc.)

    cilt tipi: Skin type for which the product is labeled suitable

    içerik: One of 45 ingredients

    is_exist: Whether that ingredient exists in that product (1/0)

Ingredient presence info was gathered by querying ChatGPT and DeepSeek with product names.

🧪 Ingredient List

Each product is analyzed across a fixed list of 45 ingredients, including:

Water, Glycerin, Niacinamide, Hyaluronic Acid, Panthenol, Aloe Vera Extract, Vitamin E, Citric Acid, Phenoxyethanol, Fragrance, ...

(See full list in the products_dataset.csv)

🧹 Data Preprocessing

    The dataset was pivoted into wide format, creating one row per (product, category, skin type) and 45 ingredient columns.

    Rows with mixed skin types (e.g., "KURU-YAĞLI") were excluded from model training to maintain label purity.

    One-hot columns for each skin type were added for clarity (not used in training).

    Ingredient presence was treated as binary features (0/1).

    For YAĞLI skin type, additional KMeans clustering was used to prune potentially misclassified samples (by checking distance to KURU centroid).

🧠 Model Training Details

For each skin type:

    Positive samples = rows labeled only with that skin type.

    Negative samples = rows labeled with any other single skin type.

    Data was split using stratified 80/20 split.

    SMOTE was applied to balance training classes.

    Initial model trained with all features.

    SHAP analysis used to select top 15 features for YAĞLI and KARMA skin types.

    Final model retrained using selected features.

    Model and SHAP explainer saved to ./models/model_{skin}.pkl and explainer_{skin}.pkl.

📈 Evaluation

    Evaluation was performed on the 20% test set.

    Outputs include:

        Classification report

        ROC-AUC score

        Sample SHAP explanation for selected product

🔬 Explainability with SHAP

The model includes a built-in explain_product() method, which:

    Predicts suitability score for a given product–skin type pair

    Prints top 5 SHAP drivers for that prediction, showing which ingredients increase or decrease the suitability

Example output:

Explanation for 'CeraVe Moisturizing Cream' on model 'KURU':
 • Predicted probability of Suitable = 0.89 → Label: Suitable
 • Top SHAP drivers:
   - Glycerin           : SHAP = +0.35 (↑ Suitable)
   - Alcohol Denat      : SHAP = -0.27 (↓ Not Suitable)
   ...


Each file would handle training + saving for one skin type model, avoiding confusion and ensuring reproducibility.
