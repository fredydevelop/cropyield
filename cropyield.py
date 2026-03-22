#Importing the dependencies
import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import streamlit as st
import base64
import pickle as pk
from streamlit_option_menu import option_menu



data_header_names = [
    'Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used',
    'Irrigation_Used', 'Days_to_Harvest', 'Region_East', 'Region_North',
    'Region_South', 'Region_West', 'Soil_Type_Chalky', 'Soil_Type_Clay',
    'Soil_Type_Loam', 'Soil_Type_Peaty', 'Soil_Type_Sandy',
    'Soil_Type_Silt', 'Crop_Barley', 'Crop_Cotton', 'Crop_Maize',
    'Crop_Rice', 'Crop_Soybean', 'Crop_Wheat',
    'Weather_Condition_Cloudy', 'Weather_Condition_Rainy',
    'Weather_Condition_Sunny'
]

st.set_page_config(page_title='Crop Yield Prediction System', layout='centered')

st.image('logo.jpeg', width=120, caption='AgroRegressor')
selection = option_menu(
    menu_title=None,
    options=["Single Prediction", "Multi Prediction"],
    icons=["cast", "book"],
    default_index=0,
    orientation="horizontal"
)


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href


def cropYield(givendata):
    loaded_model = pk.load(open("The_Latest_cropYield_Model.sav", "rb"))
    std_scaler_loaded = pk.load(open("my_saved_std_scaler.pkl", "rb"))

    input_data_as_numpy_array = np.asarray(givendata, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_X_resample = std_scaler_loaded.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_X_resample)
    predicted_yield = prediction[0]

    return f"Expected crop yield: {predicted_yield:.2f}"



def main():
    st.header("Agricultural Crop Yield Prediction System")

    Rainfall_mm = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        placeholder="Enter rainfall amount in millimeters"
    )

    Temperature_Celsius = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        placeholder="Enter temperature in Celsius"
    )

    Days_to_Harvest = st.number_input(
        "Days to Harvest",
        min_value=0,
        step=1
    )

    soil_type = st.selectbox(
        "Enter the soil type",
        ("Select option", "Sandy", "Clay", "Loam", "Silt", "Peaty", "Chalky"),
        key="soil_type"
    )

    crop_type = st.selectbox(
        "Enter the crop type",
        ("Select option", "Cotton", "Rice", "Barley", "Soybean", "Wheat", "Maize"),
        key="crop_type"
    )

    fert_option = st.selectbox(
        "Is fertilizer used?",
        ("Select option", "Yes", "No"),
        key="fert_used"
    )

    irrig_option = st.selectbox(
        "Is irrigation used?",
        ("Select option", "Yes", "No"),
        key="irrig_used"
    )

    Weather_Condition = st.selectbox(
        "What type of weather condition?",
        ("Select option", "Cloudy", "Rainy", "Sunny"),
        key="weather_cond"
    )

    Region = st.selectbox(
        "Where is the farm located?",
        ("Select option", "West", "South", "North", "East"),
        key="region"
    )

    # Initialize one-hot variables inside main
    Soil_Type_Chalky = 0
    Soil_Type_Clay = 0
    Soil_Type_Loam = 0
    Soil_Type_Peaty = 0
    Soil_Type_Sandy = 0
    Soil_Type_Silt = 0

    Crop_Barley = 0
    Crop_Cotton = 0
    Crop_Maize = 0
    Crop_Rice = 0
    Crop_Soybean = 0
    Crop_Wheat = 0

    Weather_Condition_Cloudy = 0
    Weather_Condition_Rainy = 0
    Weather_Condition_Sunny = 0

    Region_East = 0
    Region_North = 0
    Region_South = 0
    Region_West = 0

    # Encode soil type
    if soil_type == 'Sandy':
        Soil_Type_Sandy = 1
    elif soil_type == 'Clay':
        Soil_Type_Clay = 1
    elif soil_type == 'Loam':
        Soil_Type_Loam = 1
    elif soil_type == 'Silt':
        Soil_Type_Silt = 1
    elif soil_type == 'Peaty':
        Soil_Type_Peaty = 1
    elif soil_type == 'Chalky':
        Soil_Type_Chalky = 1

    # Encode crop type
    if crop_type == 'Cotton':
        Crop_Cotton = 1
    elif crop_type == 'Rice':
        Crop_Rice = 1
    elif crop_type == 'Barley':
        Crop_Barley = 1
    elif crop_type == 'Soybean':
        Crop_Soybean = 1
    elif crop_type == 'Wheat':
        Crop_Wheat = 1
    elif crop_type == 'Maize':
        Crop_Maize = 1

    # Encode fertilizer and irrigation
    Fertilizer_Used = 1 if fert_option == "Yes" else 0 if fert_option == "No" else None
    Irrigation_Used = 1 if irrig_option == "Yes" else 0 if irrig_option == "No" else None

    # Encode weather
    if Weather_Condition == 'Cloudy':
        Weather_Condition_Cloudy = 1
    elif Weather_Condition == 'Rainy':
        Weather_Condition_Rainy = 1
    elif Weather_Condition == 'Sunny':
        Weather_Condition_Sunny = 1

    # Encode region
    if Region == 'West':
        Region_West = 1
    elif Region == 'South':
        Region_South = 1
    elif Region == 'North':
        Region_North = 1
    elif Region == 'East':
        Region_East = 1

    if st.button("Predict Yield"):
        if (
            soil_type == "Select option" or
            crop_type == "Select option" or
            fert_option == "Select option" or
            irrig_option == "Select option" or
            Weather_Condition == "Select option" or
            Region == "Select option"
        ):
            st.error("Please fill in all fields before prediction.")
        else:
            input_data = [
                Rainfall_mm, Temperature_Celsius, Fertilizer_Used,
                Irrigation_Used, Days_to_Harvest, Region_East, Region_North,
                Region_South, Region_West, Soil_Type_Chalky, Soil_Type_Clay,
                Soil_Type_Loam, Soil_Type_Peaty, Soil_Type_Sandy,
                Soil_Type_Silt, Crop_Barley, Crop_Cotton, Crop_Maize,
                Crop_Rice, Crop_Soybean, Crop_Wheat,
                Weather_Condition_Cloudy, Weather_Condition_Rainy,
                Weather_Condition_Sunny
            ]

            result = cropYield(input_data)
            st.success(result)



# def multi(input_data):
#     loaded_model=pk.load(open("The_Latest_cropYield_Model.sav", "rb"))
#     dfinput = pd.read_csv(input_data)
#     # dfinput=dfinput.iloc[1:].reset_index(drop=True)

#     st.header('A view of the uploaded dataset')
#     st.markdown('')
#     st.dataframe(dfinput)

#     dfinput=dfinput.values
#     std_scaler_loaded=pk.load(open("my_saved_std_scaler.pkl", "rb"))
#     std_dfinput=std_scaler_loaded.transform(dfinput)
    
    
#     predict=st.button("predict")


#     if predict:
#         prediction = loaded_model.predict(std_dfinput)
#         interchange=[]
#         for i in prediction:
#             if i==1:
#                 newi="cropYield Detected"
#                 interchange.append(newi)
#             elif i==0:
#                 newi="No cropYield"
#                 interchange.append(newi)
            
#         st.subheader('Here is your prediction')
#         prediction_output = pd.Series(interchange, name='Diabetics results')
#         prediction_id = pd.Series(np.arange(len(interchange)),name="Patient_ID")
#         dfresult = pd.concat([prediction_id, prediction_output], axis=1)
#         st.dataframe(dfresult)
#         st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

def multi(input_data):
    # Load saved objects
    loaded_model = pk.load(open("crop_yield_model.pkl", "rb"))
    std_scaler_loaded = pk.load(open("crop_yield_scaler.pkl", "rb"))
    loaded_encoder = pk.load(open("crop_yield_encoder.pkl", "rb"))  # you must save this during training

    # Read uploaded CSV
    dfinput = pd.read_csv(input_data)

    st.header("A view of the uploaded dataset")
    st.dataframe(dfinput)

    # Required raw input columns
    required_columns = [
        "Rainfall_mm",
        "Temperature_Celsius",
        "Fertilizer_Used",
        "Irrigation_Used",
        "Days_to_Harvest",
        "Region",
        "Soil_Type",
        "Crop",
        "Weather_Condition"
    ]

    missing_cols = [col for col in required_columns if col not in dfinput.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    # Separate categorical and numerical columns
    categorical_columns = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
    numerical_columns = [
        "Rainfall_mm",
        "Temperature_Celsius",
        "Fertilizer_Used",
        "Irrigation_Used",
        "Days_to_Harvest"
    ]

    # Encode categoricals
    encoded_array = loaded_encoder.transform(dfinput[categorical_columns])
    encoded_feature_names = loaded_encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=dfinput.index)

    # Combine with numerical columns
    final_df = pd.concat([dfinput[numerical_columns], encoded_df], axis=1)

    # Reorder columns to exactly match training order
    data_header_names = [
        'Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used',
        'Irrigation_Used', 'Days_to_Harvest', 'Region_East', 'Region_North',
        'Region_South', 'Region_West', 'Soil_Type_Chalky', 'Soil_Type_Clay',
        'Soil_Type_Loam', 'Soil_Type_Peaty', 'Soil_Type_Sandy',
        'Soil_Type_Silt', 'Crop_Barley', 'Crop_Cotton', 'Crop_Maize',
        'Crop_Rice', 'Crop_Soybean', 'Crop_Wheat',
        'Weather_Condition_Cloudy', 'Weather_Condition_Rainy',
        'Weather_Condition_Sunny'
    ]

    # Add any missing encoded columns as 0
    for col in data_header_names:
        if col not in final_df.columns:
            final_df[col] = 0

    final_df = final_df[data_header_names]

    # Scale
    std_dfinput = std_scaler_loaded.transform(final_df)

    if st.button("Predict"):
        prediction = loaded_model.predict(std_dfinput)

        result_df = dfinput.copy()
        result_df["Predicted_Crop_Yield"] = prediction

        st.subheader("Here is your prediction")
        st.dataframe(result_df)

        st.markdown(filedownload(result_df), unsafe_allow_html=True)


if selection =="Single Prediction":
    main()

if selection == "Multi Prediction":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #---------------------------------#
    # Prediction
    #--------------------------------
    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.header('Upload your csv file here')
    uploaded_file = st.file_uploader("", type=["csv"])
    #--------------Visualization-------------------#
    # Main panel
    
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file).
        multi(uploaded_file)
    else:
        st.info('Upload your dataset !!')
    
    
