import joblib
def predict(data):
    clf = joblib.load("medical-fraud-detection-analysis-main/cb_modeltop5.joblib")
    return clf.predict_proba(data)[0][0]
