# ============================================================
#  Heart Disease Prediction - Streamlit Web App
#  ออกแบบสวยงามแบบเรียบง่าย
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============================================================
# ตั้งค่าหน้าเว็บ
# ============================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - ออกแบบเรียบง่ายและทันสมัย
st.markdown("""
<style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* หัวข้อหลัก */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        text-align: center;
        border-left: 4px solid #3b82f6;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3a8a;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
    
    /* ผลลัพธ์ */
    .result-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3);
        margin: 1rem 0;
    }
    
    .result-warning {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(239, 68, 68, 0.3);
        margin: 1rem 0;
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .result-confidence {
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
    
    /* ปุ่ม */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Container */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# โหลดโมเดล
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

@st.cache_resource
def load_metadata():
    return joblib.load('model_metadata.pkl')

try:
    model = load_model()
    metadata = load_metadata()
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {str(e)}")
    st.info("💡 ตรวจสอบว่ามีไฟล์ `heart_disease_model.pkl` และ `model_metadata.pkl` อยู่ในโฟลเดอร์")
    st.stop()

# ============================================================
# Header
# ============================================================
st.markdown('<div class="main-title">🫀 Heart Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ระบบทำนายความเสี่ยงโรคหัวใจด้วย Machine Learning</div>', unsafe_allow_html=True)

# ============================================================
# Sidebar - ข้อมูลโมเดล
# ============================================================
with st.sidebar:
    st.markdown("### 📊 ข้อมูลโมเดล")
    
    # Metrics cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata['accuracy']*100:.1f}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata['roc_auc']*100:.1f}%</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box" style="margin-top: 1rem;">
        <b>🔬 Algorithm:</b> Decision Tree<br>
        <b>📈 Precision:</b> {metadata['precision']*100:.1f}%<br>
        <b>🎯 Recall:</b> {metadata['recall']*100:.1f}%<br>
        <b>⚖️ F1-Score:</b> {metadata['f1']*100:.1f}%
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ คำอธิบาย Features")
    
    feature_info = {
        'Age': 'อายุ (ปี)',
        'Sex': 'เพศ (0=หญิง, 1=ชาย)',
        'ChestPainType': 'อาการเจ็บหน้าอก (0-4)',
        'RestingBP': 'ความดันโลหิตขณะพัก (mm Hg)',
        'Cholesterol': 'คอเลสเตอรอล (mg/dl)',
        'FastingBS': 'น้ำตาล > 120 mg/dl',
        'RestingECG': 'ผล ECG ขณะพัก (0-2)',
        'MaxHR': 'อัตราการเต้นหัวใจสูงสุด',
        'ExerciseAngina': 'เจ็บหน้าอกเมื่อออกกำลังกาย',
        'Oldpeak': 'ST depression',
        'ST_Slope': 'ความชัน ST segment'
    }
    
    for feat, desc in feature_info.items():
        st.markdown(f"**{feat}:** {desc}")

# ============================================================
# Main Content - ฟอร์มกรอกข้อมูล
# ============================================================
st.markdown("### 📝 กรอกข้อมูลสุขภาพ")

with st.form("prediction_form"):
    # ส่วนที่ 1: ข้อมูลพื้นฐาน
    st.markdown("#### 👤 ข้อมูลพื้นฐาน")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("อายุ (ปี)", min_value=20, max_value=100, value=50)
        sex = st.selectbox("เพศ", [0, 1],
                          format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)")
    
    with col2:
        chest_pain = st.selectbox("อาการเจ็บหน้าอก", [0, 1, 2, 3, 4],
                                 format_func=lambda x: {
                                     0: "0 - Typical Angina",
                                     1: "1 - Atypical Angina",
                                     2: "2 - Non-anginal Pain",
                                     3: "3 - Asymptomatic",
                                     4: "4 - Unknown"
                                 }[x])
        resting_bp = st.number_input("ความดันโลหิต (mm Hg)",
                                    min_value=80, max_value=200, value=120)
    
    with col3:
        cholesterol = st.number_input("คอเลสเตอรอล (mg/dl)",
                                     min_value=100, max_value=600, value=200)
        fasting_bs = st.selectbox("น้ำตาลในเลือด > 120 mg/dl", [0, 1],
                                 format_func=lambda x: "ใช่ (1)" if x == 1 else "ไม่ใช่ (0)")
    
    with col4:
        resting_ecg = st.selectbox("ผล ECG ขณะพัก", [0, 1, 2],
                                  format_func=lambda x: {
                                      0: "0 - Normal",
                                      1: "1 - ST-T abnormality",
                                      2: "2 - LV hypertrophy"
                                  }[x])
        max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด",
                                min_value=60, max_value=220, value=140)
    
    st.markdown("---")
    
    # ส่วนที่ 2: ข้อมูลการออกกำลังกาย
    st.markdown("#### ❤️ ข้อมูลการตรวจขณะออกกำลังกาย")
    col5, col6, col7 = st.columns(3)
    
    with col5:
        exercise_angina = st.selectbox("เจ็บหน้าอกเมื่อออกกำลังกาย", [0, 1],
                                      format_func=lambda x: "ใช่ (1)" if x == 1 else "ไม่ใช่ (0)")
    
    with col6:
        oldpeak = st.number_input("ST depression (Oldpeak)",
                                 min_value=-3.0, max_value=10.0, value=1.0, step=0.1)
    
    with col7:
        st_slope = st.selectbox("ความชัน ST segment", [0, 1, 2],
                               format_func=lambda x: {
                                   0: "0 - Upsloping",
                                   1: "1 - Flat",
                                   2: "2 - Downsloping"
                               }[x])
    
    st.markdown("---")
    
    # ปุ่มทำนาย
    submitted = st.form_submit_button("🔍 ทำนายผล", use_container_width=True)

# ============================================================
# แสดงผล
# ============================================================
if submitted:
    # สร้าง DataFrame จากข้อมูลที่กรอก
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    # ทำนายผล
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    st.markdown("### 🎯 ผลการทำนาย")
    
    # แสดงผลลัพธ์
    if prediction == 1:
        st.markdown(f"""
        <div class="result-warning">
            <div class="result-title">⚠️ มีความเสี่ยงเป็นโรคหัวใจ</div>
            <div class="result-confidence">ความมั่นใจ: {probability[1]*100:.1f}%</div>
            <p style="margin-top: 1rem;">💡 <b>คำแนะนำ:</b> ควรปรึกษาแพทย์เพื่อตรวจเพิ่มเติม</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-success">
            <div class="result-title">✅ ไม่พบความเสี่ยงโรคหัวใจ</div>
            <div class="result-confidence">ความมั่นใจ: {probability[0]*100:.1f}%</div>
            <p style="margin-top: 1rem;">💡 <b>คำแนะนำ:</b> รักษาสุขภาพและตรวจสุขภาพประจำปี</p>
        </div>
        """, unsafe_allow_html=True)
    
    # แสดงกราฟความน่าจะเป็น
    st.markdown("### 📊 ความน่าจะเป็น")
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['ไม่มีโรคหัวใจ', 'มีโรคหัวใจ']
    colors = ['#10b981', '#ef4444']
    bars = ax.bar(categories, probability, color=colors, edgecolor='white',
                  linewidth=2, width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('ความน่าจะเป็น', fontsize=12)
    ax.set_title('Prediction Probability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars, probability):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{prob*100:.1f}%', ha='center', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # แสดงข้อมูลสรุป
    st.markdown("### 📋 สรุปข้อมูลที่ใช้ทำนาย")
    st.dataframe(input_data.T, use_container_width=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    <p>🫀 Heart Disease Prediction System | Powered by Decision Tree & Scikit-learn</p>
    <p style='font-size: 0.85rem;'>⚠️ ผลลัพธ์นี้เป็นเพียงการคาดการณ์เบื้องต้น ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้</p>
</div>
""", unsafe_allow_html=True)