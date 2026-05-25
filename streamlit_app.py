"""
Customer Churn Prediction — Streamlit Web App
Запуск: streamlit run streamlit_app.py
"""

import pickle

import pandas as pd
import streamlit as st

# ── Конфигурация страницы ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# ── Кастомные стили ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main { background: #0f1117; }

    .hero {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 32px 40px;
        margin-bottom: 28px;
    }
    .hero h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        color: #e2e8f0;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero p { color: #64748b; margin: 0; font-size: 0.95rem; }

    .result-churn {
        background: linear-gradient(135deg, #2d1515, #3d1a1a);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 28px 36px;
        text-align: center;
    }
    .result-stay {
        background: linear-gradient(135deg, #0f2d1a, #0f3320);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 28px 36px;
        text-align: center;
    }
    .result-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0 0 8px 0;
    }
    .result-sub { color: #94a3b8; font-size: 0.9rem; margin: 0; }

    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    .section-header {
        font-size: 0.78rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 12px 0;
        border-bottom: 1px solid #1e2433;
        padding-bottom: 6px;
    }

    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        cursor: pointer;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #2563eb; }

    .risk-bar-bg {
        background: #1e2433;
        border-radius: 6px;
        height: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Загрузка модели ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()


# ── Заголовок ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>📊 Customer Churn Prediction</h1>
        <p>Система прогнозирования оттока клиентов e-commerce · XGBoost · AUC-ROC 0.99</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Лейаут: форма слева, результат справа ────────────────────────────────────
col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown('<div class="section-header">Профиль клиента</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        tenure = st.number_input("Tenure (мес.)", min_value=0, max_value=61, value=5)
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
        hour_spend = st.number_input("Часов в приложении/день", min_value=0, max_value=5, value=3)
        num_devices = st.number_input("Устройств зарегистрировано", min_value=1, max_value=6, value=2)
        satisfaction = st.slider("Оценка удовлетворённости", 1, 5, 3)
        num_address = st.number_input("Число адресов", min_value=1, max_value=22, value=2)
        complain = st.selectbox("Жалоба", [0, 1], format_func=lambda x: "Нет" if x == 0 else "Есть")

    with c2:
        login_device = st.selectbox("Устройство входа", ["Mobile Phone", "Computer", "Phone"])
        payment_mode = st.selectbox(
            "Способ оплаты",
            ["Debit Card", "Credit Card", "CC", "UPI", "E wallet", "COD", "Cash on Delivery"],
        )
        gender = st.selectbox("Пол", ["Male", "Female"])
        order_cat = st.selectbox(
            "Категория заказов",
            ["Laptop & Accessory", "Mobile Phone", "Mobile", "Fashion", "Grocery", "Others"],
        )
        marital_status = st.selectbox("Семейное положение", ["Single", "Married", "Divorced"])
        warehouse_dist = st.number_input("Склад → дом (км)", min_value=5, max_value=127, value=10)

    st.markdown('<div class="section-header">Активность и покупки</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        order_hike = st.number_input("Рост заказов vs прошлый год, %", min_value=11, max_value=26, value=11)
        coupon_used = st.number_input("Купонов использовано", min_value=0, max_value=16, value=1)
    with c4:
        order_count = st.number_input("Заказов всего", min_value=1, max_value=16, value=2)
        days_since_order = st.number_input("Дней с последнего заказа", min_value=0, max_value=46, value=5)

    cashback = st.number_input("Кэшбэк (руб.)", min_value=0, max_value=325, value=150)

    st.write("")
    predict_btn = st.button("⚡ Предсказать")

# ── Результат ────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-header">Прогноз</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            "Tenure": tenure,
            "PreferredLoginDevice": login_device,
            "CityTier": city_tier,
            "WarehouseToHome": warehouse_dist,
            "PreferredPaymentMode": payment_mode,
            "Gender": gender,
            "HourSpendOnApp": hour_spend,
            "NumberOfDeviceRegistered": num_devices,
            "PreferedOrderCat": order_cat,
            "SatisfactionScore": satisfaction,
            "MaritalStatus": marital_status,
            "NumberOfAddress": num_address,
            "Complain": complain,
            "OrderAmountHikeFromlastYear": order_hike,
            "CouponUsed": coupon_used,
            "OrderCount": order_count,
            "DaySinceLastOrder": days_since_order,
            "CashbackAmount": float(cashback),
        }])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        churn_prob = proba[1] * 100
        stay_prob = proba[0] * 100

        # Основной вердикт
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-churn">
                    <p class="result-label" style="color:#ef4444;">⚠ Клиент уйдёт</p>
                    <p class="result-sub">Высокий риск оттока — рекомендуется удерживающее предложение</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-stay">
                    <p class="result-label" style="color:#22c55e;">✓ Клиент останется</p>
                    <p class="result-sub">Низкий риск оттока — стандартный режим взаимодействия</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Метрики вероятностей
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Вероятность оттока</div>
                    <div class="metric-value" style="color:#ef4444;">{churn_prob:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Вероятность удержания</div>
                    <div class="metric-value" style="color:#22c55e;">{stay_prob:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Бар риска
        st.write("")
        st.markdown("**Уровень риска оттока**")
        st.progress(int(churn_prob))

        # Интерпретация
        st.write("")
        st.markdown('<div class="section-header">Рекомендации</div>', unsafe_allow_html=True)

        if churn_prob >= 70:
            st.error("🔴 **Критический риск.** Персональный звонок менеджера + эксклюзивный оффер.")
        elif churn_prob >= 40:
            st.warning("🟡 **Средний риск.** Email-кампания с купоном или повышенным кэшбэком.")
        else:
            st.success("🟢 **Низкий риск.** Поддерживать текущий уровень обслуживания.")

        # Факторы риска
        if complain == 1:
            st.info("💬 Клиент подавал жалобу — приоритет для сервисной команды.")
        if tenure < 3:
            st.info("🕐 Клиент новый (Tenure < 3 мес.) — повышенный риск ранней отписки.")
        if days_since_order > 15:
            st.info("📦 Давно не делал заказов — триггер для реактивационной кампании.")
        if satisfaction <= 2:
            st.info("⭐ Низкая оценка удовлетворённости — узнать причину через NPS-опрос.")

    else:
        st.markdown(
            """
            <div style="
                border: 1px dashed #2a2f3e;
                border-radius: 12px;
                padding: 60px 30px;
                text-align: center;
                color: #475569;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">📋</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem;">
                    Заполните форму и нажмите<br>«Предсказать»
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Информация о модели
    st.write("")
    st.markdown('<div class="section-header">О модели</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#475569; font-size:0.85rem; line-height:1.7;">
        <b style="color:#64748b;">Алгоритм:</b> XGBoost Classifier<br>
        <b style="color:#64748b;">Предобработка:</b> Median Imputer + OneHotEncoder<br>
        <b style="color:#64748b;">Метрика:</b> AUC-ROC = 0.9927 на hold-out тесте<br>
        <b style="color:#64748b;">Датасет:</b> 5 630 клиентов e-commerce (Kaggle)<br>
        <b style="color:#64748b;">Признаков:</b> 18 (поведенческие + демографические)
        </div>
        """,
        unsafe_allow_html=True,
    )
