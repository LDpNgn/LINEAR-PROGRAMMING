import streamlit as st
import pandas as pd
import numpy as np
from qhtt_oop import LinearModel


st.set_page_config(
    page_title="input"
)
st.title("Nhập bài toán Quy Hoạch Tuyến Tính")
st.write(""" 
    #### input
    - A (dataframe): lưu ràng buộc đẳng thức/bất đẳng thức
    - B (dataframe): ràng buộc về dấu của biến
    """)

def matrix_to_df(df):
    lines = df.split('\n')
    data = []
    for line in lines:
        row = []
        for value in line.split(','):
            value = value.strip()
            if value in ['>=', '<=', '=', 'min', 'max']:
                row.append(value)
            elif value == '':
                row.append(np.nan)
            else:
                row.append(float(value))
        data.append(row)
    return pd.DataFrame(data)

st.subheader("Nhập ma trận A (ràng buộc đẳng thức/bất đẳng thức):")
st.write("Ví dụ bài toán:")
st.latex(r'''z = min  (5x_1 - 10x_2)\\
                -2x_1 + x_2 <= 1\\
                x_1 - x_2 >= -2\\
                3x_1 + x_2 <= 8\\
                -2x_1 + 3x_2 >= -9\\
                4x_1 + 3x_2 >= 0''')
st.write("Ta nhập:")
st.code("""5,-10,,min  
-2,1,<=,1  
1,-1,>=,-2  
3,1,<=,8  
-2,3,>=,-9  
4,3,>=,0""")
A = st.text_area("Nhập ma trận A", value="""5,-10,,min  
-2,1,<=,1  
1,-1,>=,-2  
3,1,<=,8  
-2,3,>=,-9  
4,3,>=,0""")

st.subheader("Nhập ma trận B (ràng buộc về dấu của biến):")
st.write("Ví dụ bài toán trên có các ràng buộc sau:")
st.latex(r'''x_1 >= 0\\
            x_2 \text{ tự do}''')
st.write("Ta nhập:")
st.code(">=,0")
B = st.text_area("Nhập ma trận B", value=">=,0")

A = matrix_to_df(A)
B = matrix_to_df(B)
model = LinearModel()
model.processingInput(A, B)

st.write('Dạng chuẩn:\n')
st.write("A =\n", model.A, "\n")
st.write("b =", model.b, "\n")
st.write("c =", model.c, "\n")

model.Simplex(step_by_step=False)
st.subheader('Kết quả:')
st.write("Nghiệm tối ưu của bài toán: ", model.x)
st.write("Giá trị tối ưu của bài toán: ", round(model.z, 2))

if (st.button('Lời giải chi tiết:')):
    #st.latex(r'a \qquad  a')
    model.Step_by_step()
    


