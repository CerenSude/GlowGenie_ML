
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import pandas as pd
from database.db import db
from models.ingredients import Ingredient
from models.exist_ingredients import ExistIngredient
from ml.productsuitability_loader import ML_MODELS



def build_feature_vector(product_id, model_features):
    all_ingredients = Ingredient.query.all()
    id_to_name = {i.ingredient_id: i.ingredient_name.lower() for i in all_ingredients}

    exists = ExistIngredient.query.filter_by(product_id=product_id).all()
    ingredient_map = {id_to_name[e.ingredient_id]: e.is_exist for e in exists}

    return pd.DataFrame([[ingredient_map.get(f.lower(), 0) for f in model_features]], columns=model_features)

def predict_suitability(product_id, skin_type):
    if skin_type not in ML_MODELS:
        raise ValueError(f"No model for skin type: {skin_type}")

    model, _ = ML_MODELS[skin_type]
    features = model.feature_names_in_
    X = build_feature_vector(product_id, features)
    prob = model.predict_proba(X)[0, 1]
    label = "Suitable" if prob >= 0.3 else "Not Suitable"
    return round(float(prob), 3), label
