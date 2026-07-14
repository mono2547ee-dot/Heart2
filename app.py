# ============================================================
# app.py - Heart Disease Prediction Web App
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# ตั้งค่าหน้าเว็บ
# ============================================================
st.set_page_config(
    page_title="🫀 Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

# CSS สไตล์เรียบง่าย
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stButton > button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.7rem;
    }
    .stButton > button:hover { background-color: #c0392b; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# โหลดข้อมูล + สร้างโมเดล (อัตโนมัติ)
# ============================================================
@st.cache_resource
def build_model():
    """โหลด CSV → Preprocessing → Train โมเดล"""
    # 1. โหลดข้อมูล
    df = pd.read_csv('Heart4.csv')

    # 2. จัดการ missing values (ค่า 0 → NaN)
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)

    # 3. แยก X, y
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # 4. กำหนดประเภทคอลัมน์
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    cat_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG',
                'ExerciseAngina', 'ST_Slope']

    # 5. Preprocessor
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])

    # 6. แบ่ง Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. สร้าง + ฝึกโมเดล
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(
            criterion='gini', max_depth=6,
            min_samples_split=10, min_samples_leaf=5,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    # 8. ประเมินผล
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Disease', 'Has Disease'], output_dict=True)

    # Feature importance
    cat_enc = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
    all_features = num_cols + list(cat_enc.get_feature_names_out(cat_cols))
    importances = pipeline.named_steps['classifier'].feature_importances_

    return {
        'model': pipeline,
        'accuracy': acc,
        'confusion_matrix': cm,
        'report': report,
        'features': all_features,
        'importances': importances,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

# เรียก build_model (จะ cache ไว้ ไม่สร้างซ้ำ)
info = build_model()
model = info['model']

# ============================================================
# 🎯 หน้าเว็บ
# ============================================================
st.title("🫀 ระบบทำนายความเสี่ยงโรคหัวใจ")
st.caption("ใช้ Decision Tree Classifier | สร้างโมเดลอัตโนมัติจากข้อมูล Heart4.csv")
st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📊 ข้อมูลโมเดล")
    st.metric("Accuracy", f"{info['accuracy']*100:.1f}%")
    st.metric("ข้อมูลทั้งหมด", f"{info['total_samples']} rows")
    st.metric("Train / Test", f"{info['train_samples']} / {info['test_samples']}")

    st.divider()
    st.subheader("📌 Confusion Matrix")
    cm = info['confusion_matrix']
    cm_df = pd.DataFrame(cm,
                         index=['Actual: No', 'Actual: Yes'],
                         columns=['Pred: No', 'Pred: Yes'])
    st.dataframe(cm_df, use_container_width=True)

    st.divider()
    st.subheader("🎯 Feature Importance")
    imp_df = pd.DataFrame({
        'Feature': info['features'],
        'Score': info['importances']
    }).sort_values('Score', ascending=False).head(8)

    fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
    ax_imp.barh(imp_df['Feature'][::-1], imp_df['Score'][::-1], color='#e74c3c')
    ax_imp.set_xlabel('Importance')
    plt.tight_layout()
    st.pyplot(fig_imp)

# ---------- Main Content ----------
st.subheader("📝 กรอกข้อมูลสุขภาพ")

with st.form("input_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("🎂 อายุ (Age)", 20, 100, 50)
        sex = st.selectbox("👤 เพศ (Sex)", [1, 0],
                           format_func=lambda x: "ชาย" if x == 1 else "หญิง")
        cp = st.selectbox("💔 อาการเจ็บหน้าอก (ChestPainType)",
                          [0, 1, 2, 3],
                          format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                                  "Non-anginal Pain", "Asymptomatic"][x])

    with c2:
        rbp = st.number_input("🩺 ความดันขณะพัก (RestingBP)", 80, 200, 120)
        chol = st.number_input("🧪 คอเลสเตอรอล (Cholesterol)", 100, 600, 200)
        fbs = st.selectbox("🩸 น้ำตาล > 120 mg/dl (FastingBS)", [0, 1],
                           format_func=lambda x: "ไม่ใช่" if x == 0 else "ใช่")

    with c3:
        ecg = st.selectbox("📈 ผล ECG (RestingECG)", [0, 1, 2],
                           format_func=lambda x: ["Normal", "ST-T abnormality",
                                                   "LV hypertrophy"][x])
        mhr = st.number_input("❤️ Max Heart Rate", 60, 220, 140)
        exang = st.selectbox("🏃 เจ็บหน้าอกเมื่อออกกำลัง (ExerciseAngina)", [0, 1],
                             format_func=lambda x: "ไม่ใช่" if x == 0 else "ใช่")

    c4, c5 = st.columns(2)
    with c4:
        oldpeak = st.number_input("📉 ST Depression (Oldpeak)", 0.0, 10.0, 1.0, 0.1)
    with c5:
        slope = st.selectbox("📐 ST Slope", [0, 1, 2],
                             format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

    submitted = st.form_submit_button("🔍 ทำนายผล", use_container_width=True)

# ---------- ผลการทำนาย ----------
if submitted:
    input_df = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'ChestPainType': [cp],
        'RestingBP': [rbp], 'Cholesterol': [chol], 'FastingBS': [fbs],
        'RestingECG': [ecg], 'MaxHR': [mhr], 'ExerciseAngina': [exang],
        'Oldpeak': [oldpeak], 'ST_Slope': [slope]
    })

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.divider()
    st.subheader("🎯 ผลการทำนาย")

    if pred == 1:
        st.error(f"### ⚠️ มีความเสี่ยงเป็นโรคหัวใจ\n\n**ความมั่นใจ:** {proba[1]*100:.1f}%\n\n💊 **คำแนะนำ:** ควรปรึกษาแพทย์เพื่อตรวจเพิ่มเติม")
    else:
        st.success(f"### ✅ ไม่พบความเสี่ยงโรคหัวใจ\n\n**ความมั่นใจ:** {proba[0]*100:.1f}%\n\n🏃 **คำแนะนำ:** รักษาสุขภาพและตรวจสุขภาพประจำปี")

    # กราฟความน่าจะเป็น
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(['ไม่มีโรค ❤️', 'มีโรค 💔'], proba,
                  color=['#2ecc71', '#e74c3c'], edgecolor='black')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    for bar, v in zip(bars, proba):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v*100:.1f}%', ha='center', fontweight='bold')
    st.pyplot(fig)

# ---------- Footer ----------
st.divider()
st.caption("❤️ Heart Disease Prediction System | Powered by Decision Tree + Scikit-Learn")