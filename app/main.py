import streamlit as st
import numpy as np
import pandas as pd 
import pickle
import plotly.graph_objects as go

def get_clean_data():
    data= pd.read_csv("../bc_dataset/data.csv")
    data= data. drop(['id', 'Unnamed: 32'], axis=1)
    data['diagnosis']= data['diagnosis'].map({'M':1, 'B':0})
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data= get_clean_data()

    slider_labels = [
        ("radius_mean", "Radius (Mean)"),
        ("texture_mean", "Texture (Mean)"),
        ("perimeter_mean", "Perimeter (Mean)"),
        ("area_mean", "Area (Mean)"),
        ("smoothness_mean", "Smoothness (Mean)"),
        ("compactness_mean", "Compactness (Mean)"),
        ("concavity_mean", "Concavity (Mean)"),
        ("concave points_mean", "Concave Points (Mean)"),
        ("symmetry_mean", "Symmetry (Mean)"),
        ("fractal_dimension_mean", "Fractal Dimension (Mean)"),

        ("radius_se", "Radius (SE)"),
        ("texture_se", "Texture (SE)"),
        ("perimeter_se", "Perimeter (SE)"),
        ("area_se", "Area (SE)"),
        ("smoothness_se", "Smoothness (SE)"),
        ("compactness_se", "Compactness (SE)"),
        ("concavity_se", "Concavity (SE)"),
        ("concave points_se", "Concave Points (SE)"),
        ("symmetry_se", "Symmetry (SE)"),
        ("fractal_dimension_se", "Fractal Dimension (SE)"),

        ("radius_worst", "Radius (Worst)"),
        ("texture_worst", "Texture (Worst)"),
        ("perimeter_worst", "Perimeter (Worst)"),
        ("area_worst", "Area (Worst)"),
        ("smoothness_worst", "Smoothness (Worst)"),
        ("compactness_worst", "Compactness (Worst)"),
        ("concavity_worst", "Concavity (Worst)"),
        ("concave points_worst", "Concave Points (Worst)"),
        ("symmetry_worst", "Symmetry (Worst)"),
        ("fractal_dimension_worst", "Fractal Dimension (Worst)")
    ]

    input_dict={}
    for key,label in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value= float(0),
            max_value= float(data[key].max()),
            value= float(data[key].mean())
        )

    return input_dict

def get_scaled_val(input_dict):
    data= get_clean_data()
    X= data.drop(['diagnosis'],axis=1)

    scaled_dict={}

    for key,value in input_dict.items():
        max_val= X[key].max()
        min_val= X[key].min()

        scaled_val= (value - min_val) / (max_val - min_val)
        scaled_dict[key]= scaled_val

    return scaled_dict

def get_radar_chart(input_data):

    input_data= get_scaled_val(input_data)

    categories = ['Radius','Texture','Perimeter','Area',
                   'Smoothness','Compactness',
                   'Concavity', 'Concave points',
                   'Symmetry','Fractal dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['radius_mean'],input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['radius_se'],input_data['texture_se'], input_data['perimeter_se'],
            input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['radius_worst'],input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            angularaxis=dict(
                categoryorder='array',
                categoryarray=categories 
            )
        ),
        showlegend=True
    )
    return fig
    
def get_predictions(input_data):
    model= pickle.load(open("../Model/model.pkl", "rb"))
    scaler= pickle.load(open("../Model/scaler.pkl", "rb"))

    input_arr= np.array(list(input_data.values())).reshape(1,-1)
    input_arr_scaled= scaler.transform(input_arr)

    predictions= model.predict(input_arr_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is predicted to be: ")

    if predictions[0]==1:
        st.write("<span style='color:red'>Malignant</span>", unsafe_allow_html=True)
    else:
        st.write("<span style='color:green'>Benign</span>", unsafe_allow_html=True)

    st.write("Probability of it being benign: ", model.predict_proba(input_arr_scaled)[0][0])
    st.write("Probability of it being malicious: ", model.predict_proba(input_arr_scaled)[0][1])

def main():
    st.set_page_config(
        page_title="Breast cancer predictor",
        page_icon=":female-doctor:",
        initial_sidebar_state="expanded",
        layout="wide"
    )

    with open("../assests/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    with st.container():
        st.title("Breast cancer predictor")
        st.write("This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurement from the dataset. Youcan also update  the measurements by hand using the sliders  in the sidebar")

    input_data=add_sidebar()
    
    col1,col2= st.columns([4,1])

    with col1:
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        with st.container(key="prediction_box"):
            get_predictions(input_data)
    

if __name__ =='__main__':
    main()