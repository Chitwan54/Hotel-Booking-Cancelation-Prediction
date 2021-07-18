# import libraries
import numpy as np
import streamlit as st
import pickle
import pandas as pd

# Open the pickle file
file_path = r"C:\Users\Chitwan Manchanda\Desktop\Delhivery_Assignment\Best Model\LightGBM_Classifier.pkl"
pickle_in = open(file_path, 'rb')
classifier = pickle.load(pickle_in)

# Predict the chances of getting the booking canceled
def predict_booking_cancellation(features):
    
    # Predict the probabiliities based on the features
    prob = classifier.predict_proba(np.array(features).reshape(1,-1))
    
    # Return the probability(chances) of the bookings being canceled
    return np.round(prob[0][1]*100, 2)

# Function for the front end of the webpage
def web_page():
    st.title("Delhivery Assignment")
    
    # use the html temp
    html_temp = r"""
    <div style = "background-color : tomato;padding ; 10px">
    <h2 style="color:white;text-align:center;"> Hotel Booking Cancelation Predictor App </h2>
    </div>
    
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)
    
    features = ['hotel', 'lead_time', 'arrival_date_year', 'arrival_date_week_number',
               'arrival_date_day_of_month', 'stays_in_weekend_nights',
               'stays_in_week_nights', 'adults', 'country', 'previous_cancellations',
               'booking_changes', 'agent', 'adr', 'required_car_parking_spaces',
               'total_of_special_requests', 'reservation_status_date',
               'arrival_date_month_number', 'arrival_date', 'total_night_stays',
               'market_segment_Direct', 'market_segment_Groups',
               'market_segment_Offline TA/TO', 'market_segment_Online TA',
               'distribution_channel_Direct', 'distribution_channel_TA/TO',
               'deposit_type_Non Refund', 'customer_type_Transient',
               'customer_type_Transient-Party', 'reservation_status_No-Show']
    
    feature_list = []
    
    for feature in features:
        
        inp = st.text_input(feature)
        feature_list.append(inp)
    
    result = ""
    
    if st.button("Predict"):
        result = predict_booking_cancellation(feature_list)
    st.success("The chances of the hotel bookings being canceled is {}%".format(result))
    

if __name__ =='__main__':
    web_page()
    
    
    
        
        
        
        