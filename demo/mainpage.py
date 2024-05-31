import streamlit as st
import pandas as pd
import toop


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

st.subheader("Nhập ma trận A (ràng buộc đẳng thức/bất đẳng thức):")
optimize_direction = st.selectbox('Hướng mục tiêu',['min','max'])

num_bien = st.number_input('Số lượng biến', 1)
column_names = [f"x_{j+1}" for j in range(int(num_bien))]
muc_tieu = pd.DataFrame(columns=column_names)

# Thêm chức năng nhập dữ liệu cho người dùng
with st.form("Nhập hàm mục tiêu"):
    row_data = st.columns(num_bien)
    for i, col in enumerate(row_data):
        row_data[i] = col.text_input(f"${column_names[i]}$", value= 0)
    submit_button = st.form_submit_button("Gửi")
muc_tieu.loc[num_bien] = [float(x) if x else 0 for x in row_data]
if submit_button:
    st.write(muc_tieu)



num_rb = st.number_input('Số lượng ràng buộc', 1)
rb = [f"x_{j+1}" for j in range(int(num_bien))] + ['dấu', 'b']
rang_buoc = pd.DataFrame(columns=rb)

with st.form("Nhập ràng buộc"):
    rang_buoc = pd.DataFrame(columns=rb)
    for i in range(num_rb):
        row_data = st.columns(num_bien +2)
        for j, col in enumerate(row_data):
            if j < num_bien:
                row_data[j] = col.text_input(f"${rb[j]}._{i+1}$", value= 0)
            elif j == num_bien :
                row_data[j] = col.selectbox(f"${rb[j]}_{i+1}$", ['>=', '<=', '='])
            else:
                row_data[j] = col.text_input(f"${rb[j]}_{i+1}$", value= 0)
        
        for x in range(num_bien +2):
            if row_data[x] not in ['>=', '<=', '=']:
                row_data[x]= float(row_data[x]) if row_data[x] else 0
        rang_buoc.loc[i] = row_data
        
    submit_button = st.form_submit_button("Gửi")
if submit_button:
    st.write(rang_buoc)


st.subheader("Nhập ma trận B (ràng buộc về dấu của biến):")

dau = pd.DataFrame(columns=column_names)
with st.form("Nếu x tự do thì nhập là 0"):
    row_data = st.columns(num_bien)
    for i, col in enumerate(row_data):
        row_data[i] = col.selectbox(f"${column_names[i]}$",['>=','<=', 0])
    submit_button = st.form_submit_button("Gửi")
dau.loc[num_bien] = row_data
if submit_button:
    st.write(dau)

st.subheader("Giải:")

model = toop.LinearProgramming()

C = pd.concat([muc_tieu, rang_buoc], ignore_index=True)
C.iloc[0,num_bien] = ''
C.iloc[0,num_bien +1] = optimize_direction

st.write('Bài toán')
st.write(C)

st.write('Đáp án')
model.solving_LP_problem(C, dau, step_by_step=False)
if (st.button('Lời giải chi tiết:')):
    model.solving_LP_problem(C, dau, step_by_step=True)