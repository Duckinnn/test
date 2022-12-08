from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoModel
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    page_icon="👨‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded", #collapsed
)

st.title("Breast Cancer Diagnosis BY Transformer Model")
st.sidebar.subheader("Input")

models_list = ["--Select--", "40X", "100X", "200X", "400X"]
magnitude = st.sidebar.selectbox("Select the Magnification", models_list)

uploaded_file = st.sidebar.file_uploader(
    "Choose images to diagnose", type=["jpg", "jpeg", "png"], accept_multiple_files = True, key='key'
)

diagnosis = st.sidebar.button("Diagnose")
# if st.sidebar.button('Clear Uploaded File(s)') and 'key' in st.session_state.keys():
#     st.session_state.pop('key')
#     st.experimental_rerun()

# clearall = st.sidebar.button("Clear All")
# if clearall:
#     uploaded_file = []
#     st.experimental_rerun()
# print(uploaded_file, diagnosis, type(diagnosis),  network)
if uploaded_file != [] and diagnosis and magnitude != "--Select--":
    model_name_or_path = "Duckin/dd"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    predictions = []

    for each_file in uploaded_file:
        bytes_data = each_file.read()
        
        img = Image.open(BytesIO(bytes_data))

        inputs = feature_extractor(images=img, return_tensors="pt")
        outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        model_predicted = model.config.id2label[predicted_class_idx]
        
        predictions.append([img, each_file.name, model_predicted])
        #st.write(each_file)
        #st.write(each_file.name)
        #st.image(img)
        #st.write(model_predicted)
    
    df = pd.DataFrame(
            predictions, columns=["Image", "Name", "Result"]
            )
        
    st.dataframe(
        df.iloc[:,1:], width=1000 
    )

    # index = st.select_slider("Select the index of picture", df.index)
    # if index.on_change:
        # diagnosis = True
    # st.write(index)
    # st.image(predictions[index][0])
    #print(uploaded_file, diagnosis, network)

# "st.session_state object: ", st.session_state
