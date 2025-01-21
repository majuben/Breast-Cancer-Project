import streamlit as st
import pickle
import pandas as pd

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/free-photo/radiologist-analyzing-x-ray-healthcare-expertise-illuminated-generated-by-ai_188544-44183.jpg'); 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
        background-color: rgba(0, 0, 0, 0.4);
    }
    
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.4); 
        z-index: 1;
    }

    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        
        z-index: 1;
    }

    .stApp > div {
        position: relative;
        z-index: 2;
    }

    h1 {
        color: #2a7fba; 
    }
    label {
        color: #2a7fba; 
    }
    
    .stTextInput label, .stNumberInput label {
        color: #ffffff; 
        font-size: 34px; 
        font-weight: bold; 
    }
    
     
    .stColumn {
        padding: 0 25px;
        border-right: 4px solid #49535a; 
    }

    
    .stColumn:last-child {
        border-right: none;
    }

    
    .stColumns {
        display: flex;
        gap: 30px; 
    }
    
    h1 {
        font-size: 36px;
    }
    """,
    unsafe_allow_html=True
)


model_files = {
    "Modèle Arbre de décision": "model_arbre_decision.pkl",
    "Modèle K Nearest Neighbours": "model_knn.pkl",
    "Modèle Régression Linéaire": "model_regression_linéaire.pkl",
    "Modèle Régression Logistique": "model_regression_logistique.pkl",
    "Modèle SVM": "model_svm.pkl",
}

models = {}
for name, file in model_files.items():
    with open(file, "rb") as f:
        models[name] = pickle.load(f)

input_columns = [
    'RayonSE', 'TextureSE', 'PérimètreSE', 'SurfaceSE', 'LissageSE', 
    'CompacitéSE', 'ConcavitéSE', 'PointsConcavesSE', 'SymétrieSE', 
    'DimensionFractaleSE', 'RayonMoyen', 'TextureMoyenne', 'PérimètreMoyen', 
    'SurfaceMoyenne', 'LissageMoyen', 'CompacitéMoyenne', 'LissagePire', 
    'CompacitéPire', 'ConcavitéPire', 'PointsConcavesPires', 'SymétriePire', 
    'DimensionFractalePire', 'ConcavitéMoyenne', 'Points_ConcavesMoyens', 
    'SymétrieMoyenne', 'DimensionFractaleMoyenne', 'RayonPire', 'TexturePire', 
    'PérimètrePire', 'SurfacePire'
]

st.title("Breast cancer detector")

model_choice = st.selectbox("Choisir un model", list(models.keys()))
selected_model = models[model_choice]


cols = st.columns(4)

user_inputs = {}


for i, col in enumerate(input_columns):
    with cols[i % 4]:
        user_inputs[col] = st.number_input(f"{col}", value=0.0)

label_mapping = {0: "B", 1: "M"}


if st.button("Predict"):
    input_df = pd.DataFrame([user_inputs])
    
    prediction = selected_model.predict(input_df)
    
    prediction_label = label_mapping.get(prediction[0], "Unknown")
    
    if prediction_label == 'M':
        st.write("Prediction: Positive for cancer.")
    else:
        st.write("Prediction: Negative for cancer.")