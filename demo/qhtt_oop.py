import numpy as np
import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class LinearModel:
    # np.empty([0,0])  -> khoi tao mot mang 2 chieu rong
    def __init__(self, A=np.empty([0,0]), b=np.empty([0,0]), c=np.empty([0,0]), rsby=np.empty([0,0]), optimize_direction="min"):
        self.A = A
        self.b = b
        self.c = c
        self.optimize_direction = optimize_direction
        self.rsby = rsby
        # self.x = [float(0)] * len(c)
        self.x = None
        self.z = None
        self.type_result = None

    def setA(self, A):
        self.A = A
    def setB(self, b):
        self.b = b
    def setC(self, c):
        self.c = c
    def setR(self, rsby):
        self.rsby = rsby
    def setOptDirect(self, minmax):
        if (minmax == "min" or minmax == "max"):
            self.optimize_direction = minmax
        else:
            print("Invalid objective.")

    def printSolution(self):
        print("Nghiem toi uu cua bai toan: x =", self.x)
        print("Gia tri toi uu: z =", round(self.z, 2))
    
    def printTable(self, table):
        cstr = np.array([np.nan] * (len(self.A) + 1))
        temp = np.hstack((np.transpose([cstr]), table))

        print("   |\t", end = "")
        for i in range(0, len(self.c)):
            print("x_" + str(i + 1), end = "\t")
        for i in range(0, (len(temp[0]) - len(self.c) - 2)):
            print("w_" + str(i + 1), end = "\t")
        
        print('\n', end='')
        print('-' * 9 * (len(self.c) + len(self.A) + 1))
        print('z  |', end='')
        for i in range(0, len(temp)):
            if i != 0:
                print('   |', end='')
            for j in range(0, len(temp[0])):
                if (not (np.isnan(temp[i, j]))):
                    if (j == 0):
                        print(int(temp[i, j]), end = "\t")
                    else:
                        print(round(temp[i, j], 2), end = "\t")
                else:
                    print(end = "\t")
            print('')
            
    def processingInput(self, A, B):
        # 1. Kiểm tra hướng của hàm mục tiêu
        A.iloc[0] = A.iloc[0].fillna(0)
        optimize_direction = ''
        if A.iloc[0,-1] == 'max':
            optimize_direction = 'max'
        else:
            optimize_direction = 'min'
        A.iloc[0,-1] = 0
        # đưa cột cuối cùng của A về dạng numerical
        A[A.shape[1]-1] = pd.to_numeric(A[A.shape[1]-1])
        # nếu optimize_direction là max thì đưa hàm mục tiêu f(x) thành -f(-x)
        if optimize_direction == 'max':
            A.iloc[0,:] = -A.iloc[0,:]

        # 2. Xử lý ràng buộc về đẳng thức và bất đẳng thức
        # norm_arr là ma trận mới sau khi chuẩn hoá ma trận A. Dùng để làm biến đầu vào trong hàm xoay Bland
        A.iloc[0] = A.iloc[0].fillna(0)
        norm_arr = A.drop(columns=A.columns[A.shape[1] -1 -1]).to_numpy()
        nrow, ncol = norm_arr.shape
        for i in range(1,nrow):
            if A.iloc[i, -2] == ">=":
                norm_arr[i] = -norm_arr[i]
            elif A.iloc[i, -2] == "=":
                norm_arr[i, :-1] = -norm_arr[i, :-1]

        # 3. Xử lý ràng buộc về dấu của biến
        # rsby là hàm result by vars: dùng để kết luận biến cũ theo biến mới sau khi chạy xong thuật toán xoay Bland
        rsby = np.eye(B.shape[1])
        k = 0
        for j in range(B.shape[1]):
            if B.iloc[0, j] == "<=":
                norm_arr[:, k] = -norm_arr[:,k]
                rsby[:, k] = -rsby[:,k]
            elif B.iloc[0, j] == 0:
                new_col_norm_A = -norm_arr[:, k]
                norm_arr = np.insert(norm_arr, k+1, new_col_norm_A, axis = 1)
                new_col_rsby = -rsby[:, k]
                rsby = np.insert(rsby, k+1, new_col_rsby, axis = 1)
                k = k+1
            k = k+1

        self.setA(norm_arr[1:, :-1])
        self.setB(norm_arr[1:,-1])
        self.setC(norm_arr[0, :-1])
        self.setR(rsby)
        self.setOptDirect(optimize_direction)

    def printVariable(self, idx):
        if idx < len(self.c):
            return "x_" + str(idx + 1)
        else:
            return "w_" + str(idx - len(self.c) + 1)

    def checkTypeResult(self, arr_tvtu):
        tvtu = arr_tvtu[0, :-1]
        # lấy được trường hợp không giới nội KGN hoặc duy nhất nghiệm DNN
        type_result = self.selectInOut(arr_tvtu)
        if type_result == "TVTU":
            if np.count_nonzero(tvtu) < (arr_tvtu.shape[1] - arr_tvtu.shape[0] - 1):
                type_result = "VSN"
            else:
                type_result = "DNN"
        return type_result

    # phương thức xây dựng từ vựng xuất phát
    def generateTable(self):
        # xây dựng dòng đầu tiên của bảng, tương ứng hàm mục tiêu
        num_var = len(self.c)
        num_constraint = len(self.A)
        tab1 = np.hstack((self.c, [0] * (num_constraint + 1)))
        
        A = self.A

        if (not ((num_var + num_constraint) == len(self.A[0]))):
            I = np.identity(num_constraint)
            A = np.hstack((self.A, I))
        
        tab2 = np.hstack((A, np.transpose([self.b])))
        
        table = np.vstack((tab1, tab2))
        table = np.array(table, dtype ='float')
        return table
    
    def findEyeList(self, arr_col):
        lst = arr_col.tolist()
        for element in lst:
            if element not in [0,1]:
                return None
        if lst.count(1) == 1:
            return lst.index(1)
    
    def Simplex(self, step_by_step = True):
        # tạo danh sách df_result để lưu lại các từ vựng tại mỗi lần xoay
        df_result = list()
        table = self.generateTable()

        # thêm từ vựng ban đầu vào danh sách
        df_result.append(table)
        cur_arr = table.copy()

        # Xoay đơn hình cho đến khi tìm được từ vựng tối ưu
        in_out_ind = self.selectInOut(cur_arr)
        while in_out_ind != 'TVTU':
            if in_out_ind != "KGN":
                new_arr = self.blandRotate(cur_arr, in_out_ind)
                df_result.append(new_arr.copy())
            else:
                self.type_result = "KGN"
                break
            cur_arr = new_arr
            in_out_ind = self.selectInOut(cur_arr)

        self.type_result = self.checkTypeResult(df_result[-1])

        # tạo mảng lưu giá trị nghiệm tối ưu của (P)
        nrow, ncol = table.shape

        # 1. DNN
        if self.type_result == "DNN":
            arr_last = df_result[-1]
            result = np.zeros(arr_last.shape[1])
            result[-1] = arr_last[0, -1]
            for col in range(ncol - 1):
                ind = self.findEyeList(arr_last[:,col])
                if ind != None:
                    result[col] = arr_last[ind, -1]
                else:
                    result[col] = 0
            
            # self.x_hat là ma trận chứa nghiệm của các biến x1, ..., xn của (P')
            x_hat = result[:(len(result) - len(self.A) - 1)]
            # self.x_hat là ma trận chứa nghiệm của các biến x1, ..., xn của (P)
            self.x = self.rsby@x_hat
            if self.optimize_direction == 'min':
                self.z = result[-1]
            else:
                self.z = -result[-1]

        # 2. KGN
        elif self.type_result == "KGN":
            self.x = "Bai toan khong gioi noi"
            if self.optimize_direction == 'min':
                self.z = -np.inf
            else:
                self.z = np.inf
        # # 3. VSN
        # elif type_result == "VSN":

        # in ra các bước làm nếu step_by_step = True
        if (step_by_step == True):
            print(f"\n--- Cac buoc thuc hien cua phuong phap don hinh --- \n")
            print('Tu vung xuat phat:\n')
            for result in df_result:
                self.printTable(result)
                in_out_ind = self.selectInOut(result)
                if (in_out_ind == "TVTU"):
                    print("\nTu vung toi uu -> dung thuat toan.\n")
                    break
                print(f'\nVi tri xoay: ({in_out_ind[0]}, {in_out_ind[1]})')
                print("Bien vao: " + self.printVariable(in_out_ind[1]))
                for i in range(0, len(result[0]) - 1):
                    if (np.sum(result[0:, i]) == 1 and result[0:, i].tolist().index(1) == in_out_ind[0]):
                        print("Bien ra: " + self.printVariable(i))
                        break
                print("Tu vung moi sau khi xoay:\n")

        self.printSolution()
        return df_result

    @staticmethod
    def selectInOut(arr):
        """
        Output:
        - nếu i = None, tức là không có biến vào --> từ vựng tối ưu
        - nếu i != None, j = None, tức là có biến vào, không có biến ra --> bài toán không giới nội
        - nếu i != None, j != None, trả về chỉ số (i,j) cần tìm
        """
        ind_in = None
        ind_out = None
        min_val = np.amin(arr[0, 0:-1])
        if min_val < 0:
            ind_in = arr[0, 0:-1].tolist().index(min_val)
        else:
            return('TVTU')
        
        # Tìm tỷ lệ b/|a| nhỏ nhất để chọn biến ra
        minimum = 99999
        for i in range(1, len(arr)):
            if arr[i, ind_in] > 0:
                val = arr[i, -1] / arr[i, ind_in]
                if val < minimum:
                    minimum = val
                    ind_out = i

        # Bai toan Khong gioi noi
        if ind_in != None and ind_out == None:
            return('KGN') 
        
        # Tra ve vi tri cua bien can xoay
        return(ind_out, ind_in)

    @staticmethod
    def blandRotate(arr, rotate_point):
        nrow, ncol = arr.shape
        # nếu hệ số tại vị trí cần xoay != 1, scale nó về 1
        if(arr[rotate_point] != 1):
            arr[rotate_point[0],:] *= (1./arr[rotate_point])
            
        # tiến hành xoay dựa trên dòng chứa điểm xoay đã được scale ở bước trên
        for i in range(0, nrow):
            if(rotate_point[0] != i):
                arr[i,:] -= (arr[i, rotate_point[1]] * arr[rotate_point[0],:])
        
        return arr
    

    ## danh cho streamlit
    def stTable(self, table):
        cstr = np.array([np.nan] * (len(self.A) + 1))
        temp = np.hstack((np.transpose([cstr]), table))

        data = []
        for i in range(0, len(temp)):
            row =[]
            if i == 0:
                row.append('z')
            for j in range(0, len(temp[0])):
                if (not (np.isnan(temp[i, j]))):
                    if (j == 0):
                        row.append(int(temp[i, j]))
                    else:
                        row.append(round(temp[i, j], 2))
                else:
                    if i != 0:
                        row.append('' )
            data.append(row)
        columns = ['']  
        for i in range(0, len(self.c)):
            columns.append(f'x_{i+1}')
        for i in range(0, (len(temp[0]) - len(self.c) - 2)):
            columns.append(f'w_{i+1}')
        columns.append('=')
        df=pd.DataFrame(data, columns=columns)
        #st.write(pd.DataFrame(data, columns=columns))
        st.markdown(df.to_html(index=False), unsafe_allow_html=True)

    def Step_by_step(self):
        st.latex(r'\text{--- Các bước thực hiện của phương pháp đơn hình ---}')
        st.write('Từ vựng xuất phát:\n')
        for result in self.Simplex():
            self.stTable(result)
            in_out_ind = self.selectInOut(result)
            if (in_out_ind == "TVTU"):
                st.write("\nTừ vựng tối ưu -> dừng thuật toán.\n")
                break
            st.write(f'\nVị trí xoay: ({in_out_ind[0]}, {in_out_ind[1]})')
            st.write(f"Biến vào: ${self.printVariable(in_out_ind[1])}$")
            for i in range(0, len(result[0]) - 1):
                if (np.sum(result[0:, i]) == 1 and result[0:, i].tolist().index(1) == in_out_ind[0]):
                    st.write(f"Biến ra: ${self.printVariable(i)}$")
                    break
            st.write("Từ vựng mới sau khi xoay:\n")
