import os
import pickle

MODEL_DIR = "ml/productsuitability/models"
SKIN_TYPES = ["YAGLI", "KURU", "KARMA", "NORMAL"]

def load_models():
    models = {}
    for skin in SKIN_TYPES:
        with open(os.path.join(MODEL_DIR, f"model_{skin}.pkl"), "rb") as mf:
            model = pickle.load(mf)
        with open(os.path.join(MODEL_DIR, f"explainer_{skin}.pkl"), "rb") as ef:
            explainer = pickle.load(ef)
        models[skin] = (model, explainer)
    return models

ML_MODELS = load_models()
