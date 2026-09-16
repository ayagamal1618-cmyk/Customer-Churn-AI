import streamlit as st
import joblib
import pandas as pd
import numpy as np

# تحميل النموذج المدرب
model = joblib.load('logistic_model.pkl')

st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 نظام توقع مغادرة العملاء (Telco Customer Churn)")
st.write("أدخل بيانات العميل أدناه لمعرفة احتمالية مغادرته للخدمة بناءً على نموذج الانحدار اللوجستي.")

with st.form("churn_form"):
    st.subheader("معلومات العميل والخدمات")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        senior_citizen = st.selectbox("مواطن مسن (Senior Citizen)", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
        tenure = st.number_input("مدة الاشتراكات بالأشهر (Tenure)", min_value=0, max_value=120, value=12)
        monthly_charges = st.number_input("المصاريف الشهرية (Monthly Charges)", min_value=0.0, max_value=200.0, value=50.0)
        gender = st.selectbox("الجنس", ["أنثى (Female)", "ذكر (Male)"])
        partner = st.selectbox("هل يوجد شريك؟ (Partner)", ["لا", "نعم"])
        
    with col2:
        dependents = st.selectbox("هل يوجد معالين؟ (Dependents)", ["لا", "نعم"])
        phone_service = st.selectbox("خدمة الهاتف (Phone Service)", ["لا", "نعم"])
        multiple_lines = st.selectbox("خطوط متعددة (Multiple Lines)", ["لا", "نعم", "لا توجد خدمة هاتف"])
        internet_service = st.selectbox("خدمة الإنترنت (Internet Service)", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("حماية على الإنترنت (Online Security)", ["لا", "نعم", "لا توجد خدمة إنترنت"])
        
    with col3:
        online_backup = st.selectbox("نسخ احتياطي (Online Backup)", ["لا", "نعم", "لا توجد خدمة إنترنت"])
        device_protection = st.selectbox("حماية الجهاز (Device Protection)", ["لا", "نعم", "لا توجد خدمة إنترنت"])
        tech_support = st.selectbox("دعم فني (Tech Support)", ["لا", "نعم", "لا توجد خدمة إنترنت"])
        streaming_tv = st.selectbox("بث تلفزيوني (Streaming TV)", ["لا", "نعم", "لا توجد خدمة إنترنت"])
        streaming_movies = st.selectbox("بث أفلام (Streaming Movies)", ["لا", "نعم", "لا توجد خدمة إنترنت"])

    st.subheader("تفاصيل الدفع والعقود")
    col4, col5, col6 = st.columns(3)
    with col4:
        contract = st.selectbox("نوع العقد (Contract)", ["Month-to-month", "One year", "Two year"])
    with col5:
        paperless_billing = st.selectbox("فواتير إلكترونية (Paperless Billing)", ["لا", "نعم"])
    with col6:
        payment_method = st.selectbox("طريقة الدفع (Payment Method)", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])

    submit_button = st.form_submit_button(label="🔍 توقع النتيجة")

if submit_button:
    # تجهيز المدخلات بالترتيب والأسماء المطابقة لنموذجك
    input_data = {
        'SeniorCitizen': senior_citizen,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'gender_Male': 1 if gender == "ذكر (Male)" else 0,
        'Partner_Yes': 1 if partner == "نعم" else 0,
        'Dependents_Yes': 1 if dependents == "نعم" else 0,
        'PhoneService_Yes': 1 if phone_service == "نعم" else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == "لا توجد خدمة هاتف" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "نعم" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_No internet service': 1 if online_security == "لا توجد خدمة إنترنت" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "نعم" else 0,
        'OnlineBackup_No internet service': 1 if online_backup == "لا توجد خدمة إنترنت" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "نعم" else 0,
        'DeviceProtection_No internet service': 1 if device_protection == "لا توجد خدمة إنترنت" else 0,
        'DeviceProtection_Yes': 1 if device_protection == "نعم" else 0,
        'TechSupport_No internet service': 1 if tech_support == "لا توجد خدمة إنترنت" else 0,
        'TechSupport_Yes': 1 if tech_support == "نعم" else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == "لا توجد خدمة إنترنت" else 0,
        'StreamingTV_Yes': 1 if streaming_tv == "نعم" else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == "لا توجد خدمة إنترنت" else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == "نعم" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaperlessBilling_Yes': 1 if paperless_billing == "نعم" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }
    
    feature_names = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender_Male', 'Partner_Yes', 
        'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
        'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
        'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
        'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
        'TechSupport_No internet service', 'TechSupport_Yes', 
        'StreamingTV_No internet service', 'StreamingTV_Yes', 
        'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
        'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
        'PaymentMethod_Mailed check'
    ]
    
    df = pd.DataFrame([input_data])[feature_names]
    
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0][1]
    
    st.divider()
    st.subheader("نتيجة التحليل:")
    if prediction == 1:
        st.error(f"⚠️ تنبيه: العميل **معرض بنسبة عالية لمغادرة الشركة (Churn)**. (احتمالية المغادرة: {prediction_proba:.2%})")
    else:
        st.success(f"✅ العميل **مستمر ولن يغادر** الشركة غالباً. (احتمالية المغادرة: {prediction_proba:.2%})")