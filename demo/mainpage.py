import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    if (num_bien == 2):
        c =C.iloc[1:, -2].to_numpy()

        A = C.iloc[1:, :2].to_numpy()
        b = C.iloc[1:, -1].astype(float).to_numpy()

        for index, row in dau.iterrows():
            for col in dau.columns:
                if row[col] in ['>=', '<=']:
                    c = np.append(c, row[col])
                    b = np.append(b, 0)
                    if index == 0:
                        A = np.append(A, [1, 0])
                    else:
                        A = np.append(A, [0, 1])
        A = A.reshape(-1, 2)
        x1 = np.linspace(-5, 5, 100)
        x2 = np.linspace(-5, 5, 100)

        # Create a grid of x1 and x2 values
        X1, X2 = np.meshgrid(x1, x2)

        # Check if each point in the grid is feasible
        feasible = np.all(A @ np.column_stack((X1.ravel(), X2.ravel())).T <= b.reshape(-1, 1), axis=0)
        feasible_region = feasible.reshape(X1.shape)

        # Plot the feasible region
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(X1, X2, feasible_region, levels=[0.5, 1], colors='lightgreen', alpha=0.5)

        # Plot the constraints
        x1_grid = np.linspace(-5, 5, 100)
        x2_grid = np.linspace(-5, 5, 100)
        intersection = np.all([])
        for i, j in enumerate(A):
            if c[i] == '>=':
                intersection &= (j[0] * X1.ravel() + j[1] * X2.ravel() >= b[i])
            elif c[i] == '<=':
                intersection &= (j[0] * X1.ravel() + j[1] * X2.ravel() <= b[i])
            else:
                intersection &= (j[0] * X1.ravel() + j[1] * X2.ravel() == b[i])
            if j[0] !=0 and j[1]!= 0:
                ax.plot(x1_grid, (b[i] - j[0] * x1_grid)/j[1], linestyle='--', label=f'{j[0]}x_1 + {j[1]}x_2 {c[i]} {b[i]}')
            elif j[1] ==0:
                ax.plot(np.ones_like(x1_grid) * b[i], x1_grid, color='brown', linestyle='--', label=f'{j[0]}x_1 + {j[1]}x_2 {c[i]} {b[i]}')
            elif j[0] ==0:
                ax.plot(x1_grid, np.ones_like(x1_grid) * b[i], color='brown', linestyle='--', label=f'{j[0]}x_1 + {j[1]}x_2 {c[i]} {b[i]}')
        intersection = intersection.reshape(X1.shape)      
        
        # Plot the intersection
        ax.contourf(X1, X2, intersection, levels=[0.5, 1], colors='red', alpha=0.5)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Feasible Region for the Linear Programming Problem')
        ax.legend()
        st.pyplot(fig)


    model2 = toop.LinearProgramming()
    model2.solving_LP_problem(C, dau, step_by_step=True)
