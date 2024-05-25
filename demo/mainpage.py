import streamlit as st


st.set_page_config(
    page_title="QHTT"
)
st.title("Giải bài toán Quy Hoạch Tuyến Tính")



st.write(""" 
    #### input
    - A (dataframe): lưu ràng buộc đẳng thức/bất đẳng thức
    - B (dataframe): ràng buộc về dấu của biến 

    #### goal
    - chuyển 2 dataframe lưu ràng buộc đẳng thức/bất đẳng thức và ràng buộc về dấu của biến trong bài toán quy hoạch tuyến tính cho trước thành những ma trận có thể tính toán, xử lý được
    
    #### output
    - optimize_direction: hướng của hàm mục tiêu
    - norm_arr (np.array): ma trận của bài toán sau khi chuyển về dạng chuẩn (P')
    - tab_arr (np.array): ma trận tương ứng với dạng bảng của phương pháp đơn hình của norm_arr
    - rsby (np.array): ma trận lưu trữ cách biểu diễn nghiệm của (P) theo (P')
    """)