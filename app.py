# app.py
import streamlit as st
import requests
from datetime import date

st.title("Encumbrance Certificate Generator")
zone_mapping = {
    "Chennai": "1",
    "Chengalpattu": "15",
    "Coimbatore": "2",
    "Cuddalore": "3",
    "Madurai": "4",
    "Ramanathapuram": "13",
    "Salem": "5",
    "Tanjore": "7",
    "Thirunelveli": "8",
    "Trichy": "6",
    "Vellore": "9"
}


district_mapping = {
    "Chennai North": "20002",
    "Chennai Central": "20003",
    "Chennai South": "20001",
    "Thiruvalur": "50003"
}

sro_mapping = {
    "Royapuram": "20069:1",
    "Ambattur": "20067:1",
    "Chennai Central_Chennai North Joint I": "22308:0",
    "Chennai North Joint I": "20072:1",
    "Chennai South_Chennai North Joint I": "22307:0",
    "Konnur": "20077:1",
    "Madhavaram": "20655:1",
    "Muthialpet_Chennai North Joint I": "22309:0",
    "Sembiam": "20071:1",
    "Sowcarpet": "20073:1",
    "Thiruvottiyur": "20074:1"
}


village_mapping = {
    "Tondiarpet": "532",
}

selected_zone = st.selectbox("Zone", list(zone_mapping.keys()))
selected_district = st.selectbox("District", list(district_mapping.keys()))
selected_sro = st.selectbox("Sub Registrar Office", list(sro_mapping.keys()))
selected_village = st.selectbox("Village", list(village_mapping.keys()))

form_data = {
    "search_type": "Survey Wise",  
    "zone": zone_mapping[selected_zone],
    "district": district_mapping[selected_district],
    "ec_start_date": st.text_input("EC Period Start", "01Jan2000"),
    "ec_end_date": st.text_input("EC Period End", "15Apr2025"),
    "sro": sro_mapping[selected_sro],
    "village": village_mapping[selected_village],
    "survey_number": st.text_input("Survey Number", "95"),
    "subdivision_number": st.text_input("Subdivision Number", "4")
}


if st.button("Generate EC PDF"):
    with st.spinner("Processing..."):
        response = requests.post("http://localhost:8000/generate-ec", json=form_data)
        if response.status_code == 200:
            # Check if PDF was successfully downloaded
            if response.json().get("pdf_downloaded", False):
                st.success("PDF has been generated successfully! Please check your Downloads folder.")
            else:
                st.error("Failed to generate the PDF. Please try again.")
        else:
            st.error("Failed to generate the PDF. Please try again.")
