import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Load models and scaler
models = {
    "Logistic Regression": joblib.load("logistic_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "XGBoost": joblib.load("xgboost_model.pkl") 
}
scaler = joblib.load("scaler.pkl")

# Streamlit App Title
st.title("Bankruptcy Prediction App")
st.write("Predict the bankruptcy risk of a company using various machine learning models.")

# Dropdown to select the model
st.header("Select Algorithm")
algorithm = st.selectbox("Choose an algorithm:", list(models.keys()))

# Input Features
st.header("Input Features")

industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5, step=0.1)
management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5, step=0.1)
financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5, step=0.1)
credibility = st.slider("Credibility", 0.0, 1.0, 0.5, step=0.1)
competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5, step=0.1)
operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5, step=0.1)

# Prediction Section
if st.button("Predict"):
    # Prepare input data
    features = np.array([[industrial_risk, management_risk, financial_flexibility,
                          credibility, competitiveness, operating_risk]])
    features_scaled = scaler.transform(features)

    # Load selected model
    model = models[algorithm]

    # Make predictions
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[:, 1]

    # Display prediction
    if prediction[0] == 1:
        st.error(f"The company is at risk of bankruptcy. Probability: {probability[0]:.2f}")
    else:
        st.success(f"The company is not at risk of bankruptcy. Probability: {1 - probability[0]:.2f}")

# # Model Evaluation Section
# st.header("Model Evaluation")

# if st.button("Show Evaluation Metrics"):
#     # Display metrics for the selected model
#     st.subheader(f"Evaluation Metrics for {algorithm}")
    
#     # Assuming you have stored test data for evaluation
#     X_test_scaled = joblib.load("X_test_scaled.pkl")
#     y_test = joblib.load("y_test.pkl")
    
#     y_pred = model.predict(X_test_scaled)
#     y_proba = model.predict_proba(X_test_scaled)[:, 1]

#     # Classification Report
#     report = classification_report(y_test, y_pred, output_dict=True)
#     st.json(report)

#     # Confusion Matrix
#     st.subheader("Confusion Matrix")
#     cm = confusion_matrix(y_test, y_pred)
#     st.write(cm)
#     fig, ax = plt.subplots()
#     ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.6)
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
#     st.pyplot(fig)

#     # ROC Curve
#     st.subheader("ROC Curve")
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     roc_auc = auc(fpr, tpr)
#     fig, ax = plt.subplots()
#     ax.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
#     ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
#     ax.set_title("Receiver Operating Characteristic")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.legend()
#     st.pyplot(fig)










# import streamlit as st
# import joblib
# import numpy as np
# import os

# print("Current working directory:", os.getcwd())
# print("Files in current directory:", os.listdir())

# model = joblib.load('best_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Streamlit app
# st.title("Bankruptcy Prediction App")
# st.write("Predict whether a company is at risk of bankruptcy based on key risk factors.")

# # Input fields
# industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5, step=0.1)
# management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5, step=0.1)
# financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5, step=0.1)
# credibility = st.slider("Credibility", 0.0, 1.0, 0.5, step=0.1)
# competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5, step=0.1)
# operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5, step=0.1)

# # Buttons
# if st.button("Predict"):
#     features = np.array([[industrial_risk, management_risk, financial_flexibility, 
#                         credibility, competitiveness, operating_risk]])
#     features_scaled = scaler.transform(features)

#     # Make predictions
#     try:
#         prediction = model.predict(features_scaled)
#         probability = model.predict_proba(features_scaled)[:, 1]
#         if prediction[0] == 1:
#             st.error(f"The company is at risk of bankruptcy. Probability: {probability[0]:.2f}")
#         else:
#             st.success(f"The company is not at risk of bankruptcy. Probability: {1 - probability[0]:.2f}")
#     except Exception as e:
#         st.error(f"An error occurred during prediction: {e}")













# import joblib
# import numpy as np
# import streamlit as st

# # Load the model and scaler
# try:
#     model = joblib.load('best_model.pkl')
#     scaler = joblib.load('scaler.pkl')
#     st.write(f"Model loaded successfully. Type: {type(model)}")
#     st.write(f"Scaler loaded successfully. Type: {type(scaler)}")
# except Exception as e:
#     st.error(f"Error loading model or scaler: {e}")

# # Sample input for prediction
# try:
#     # Example input features
#     features = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
#     features_scaled = scaler.transform(features)  # Scale the features
#     prediction = model.predict(features_scaled)  # Make prediction
#     probability = model.predict_proba(features_scaled)[:, 1]

#     st.write(f"Prediction: {prediction}")
#     st.write(f"Probability: {probability}")
# except Exception as e:
#     st.error(f"An error occurred during prediction: {e}")
