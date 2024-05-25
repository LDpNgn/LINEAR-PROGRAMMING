import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Desmos"
)
st.title("Kiểm tra bài toán Quy Hoạch Tuyến Tính bằng phương pháp hình học (chỉ dành cho 2 biến)")

def render_desmos(width=800, height=600):
    desmos_url = f"https://www.desmos.com/calculator?"
    components.iframe(desmos_url, width=width, height=height)

# Hiển thị đồ thị với kích thước 800x600
render_desmos(width=900, height=500)
