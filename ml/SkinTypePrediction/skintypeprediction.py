import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from ctgan import CTGAN

#loading the dataset
data = pd.read_csv('ml/SkinTypePrediction/SkinTypePrediction_Dataset.csv')

#viewing the data to see
#print(data.head())

#to change the column names
#column = data.columns
#print(column)

#dropping useless columns
data = data.drop(columns=["Zaman damgası", "E-posta Adresi"], axis=1)
data.rename(columns= {
    "1. Please choose your skin type correctly.\n(Lütfen cilt tipinizi doğru seçiniz.)": "SkinType",
    "2. How often does your skin feel oily or shiny, especially in the T-zone (forehead, nose, and chin)?\n(Cildinizde, özellikle T bölgesinde (alın, burun ve çene) ne sıklıkla yağlı veya parlak bir his oluşuyor?)": "Q1",
    "3. Does your skin feel tight, dry, or flaky after cleansing?\n(Cildinizi temizledikten sonra gergin, kuru veya pul pul mu hissediyorsunuz?)": "Q2",
    "4. Do moisturizers make your skin feel greasy or hydrated?\n.(Nemlendiriciler cildinizde yapışkan bir his mi bırakıyor yoksa nemli mi hissettirir?)": "Q3",
    "5. Do you often experience blackheads or acne, especially in your T-zone (forehead, nose, chin)?\n(Özellikle T bölgenizde (alın, burun, çene) sıklıkla siyah nokta veya sivilce mi yaşıyor musunuz?)": "Q4",
    "6. Does your skin feel greasy or dry after waking up in the morning? (Sabah uyandığınızda cildiniz kuru mu yağlı mı hissettiriyor?)(a. Dengeli b. Kuru veya gergin c. Bazı bölgelerde normal T bölgesinde yağlı d. Genel olarak yağlı ve parlak)": "Q5",
    "7. How does your skin react to seasonal changes?\n(Cildiniz mevsim değişikliklerine nasıl tepki veriyor?)": "Q6",
    "8. Does your skin feel comfortable and balanced throughout the day, without the need for extra moisturizers or products?\n(Ekstra nemlendiricilere veya ürünlere ihtiyaç duymadan cildiniz gün boyu rahat ve dengeli hissediyor mu?)": "Q7",
    "9. How does your U-zone (cheeks and jawline) typically feel?\n(U bölgeniz (yanaklar ve çene çizgisi) genellikle nasıl hissediyor?)": "Q8"
}, inplace=True)

#normalizing and mapping the first col.
for col in data.columns:
    data[col] = data[col].str.lower().str.strip()

standard_map = {
    'oily (yağlı)': 'oily',
    'dry (kuru)': 'dry',
    'combination (karma)': 'combination'
}
data['SkinType'] = data['SkinType'].replace(standard_map)

#reordering SkinType to last column
data = data[data.columns[1:].tolist() + [data.columns[0]]]

#ctgan synth data gen func
def generate_synthetic_data(df, skin_type, n_samples):
    subset = df[df["SkinType"] == skin_type]
    model = CTGAN(epochs=300)
    model.fit(subset, discrete_columns=subset.columns.tolist())
    synthetic = model.sample(n_samples)
    return synthetic

#upscaling classes to total of 140 instances
dry_synth = generate_synthetic_data(data, 'dry', 130)
normal_synth = generate_synthetic_data(data, 'normal', 117)
oily_synth = generate_synthetic_data(data, 'oily', 112)
comb_synth = generate_synthetic_data(data, 'combination', 53)

#real+synth data
augmented_data = pd.concat([data, dry_synth, normal_synth, oily_synth, comb_synth], ignore_index=True)

#encoding features and label
X = pd.get_dummies(augmented_data.drop(columns='SkinType'))
le = LabelEncoder()
y = le.fit_transform(augmented_data['SkinType'])

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = LogisticRegression(
    solver='lbfgs',
    penalty='l2',
    C=1.0,
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


#saving the model
joblib.dump(model, 'skintypeprediction.pkl')
#sns.countplot(x='SkinType', data=data)
#plt.title('Distribution of Skin Types')
#plt.xlabel('Skin Type')
#plt.ylabel('Count')
#plt.show()


"""
log_reg = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    log_reg,
    param_grid,
    cv=5,              # 5-fold cross validation
    scoring='accuracy',
   n_jobs=-1          # use all processors
)


grid_search.fit(X, y)

print("Best C:", grid_search.best_params_['C'])
print("Best cross-validated accuracy:", grid_search.best_score_)
"""


"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print("Random Forest mean accuracy:", scores.mean())

# XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)

# Accuracy
xgb_train_acc = xgb.score(X_train, y_train)
xgb_test_acc = xgb.score(X_test, y_test)
"""