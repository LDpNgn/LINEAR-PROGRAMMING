import numpy as np
import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class LinearProgramming:
    # np.empty([0,0])  -> khoi tao mot mang 2 chieu rong
    def __init__(self, A=np.empty([0,0]), b=np.empty([0,0]), c=np.empty([0,0]), rsby=np.empty([0,0]), optimize_direction="min"):
        self.A = A
        self.b = b
        self.c = c
        self.optimize_direction = optimize_direction
        self.rsby = rsby
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
            st.write("Invalid objective.")

    def printSolution(self, step_by_step = False):
        if self.type_result == 'VSN':
            if step_by_step == True:
                st.write('Biến ở hàm mục tiêu bị khuyết so với các ràng buộc')
            st.write('Bài toán có vô số nghiệm')
            st.write('Với bổ nghiệm là: ')
            
            temp_x = []
            for element in self.x[0]:
                if element == 0:
                    temp_x.append('>=')
                else:
                    temp_x.append('=')

            temp_x = np.array(temp_x)
            cols = []
            for i in range(len(temp_x)):
                cols.append(self.printVariable(i))
            df_x0 = pd.DataFrame(temp_x.reshape(1,len(temp_x)), columns=cols)
            self.x[0] = df_x0

            cols = []
            for i in range(len(self.x[1][:, :-1])):
                cols.append(self.printVariable(i))
            df_x1 = pd.DataFrame(self.x[1][:, :-1], columns=cols, index=cols)
            df_x1 = df_x1.loc[(df_x1!=0).any(axis=1)]
            df_x1 = df_x1.loc[:, (df_x1 != 0).any(axis=0)]
            self.x[1] = df_x1

            for x in self.x:
                st.write(x)
            st.write(f"Giá trị tối ưu: {self.optimize_direction} z = {round(self.z, 2)}")
            return
        
        if self.type_result == 'KGN':
            if step_by_step == True:
                st.write('''Có vào không có ra''')
            st.write('''\nBài toán không giới nội''')
            # print('\nz: ', self.z)
            st.write(f"Giá trị tối ưu: {self.optimize_direction} z = {self.z}")
            return

        if self.type_result == 'DNN':
            st.write("Nghiệm tối ưu của bài toán: x =", self.x)
        elif self.type_result == 'VN':
            st.write("Bài toán vô nghiệm")
        st.write(f"Giá trị tối ưu: {self.optimize_direction} z = {round(self.z, 2)}")
    
    def printTable(self, table, method = 'bland'):
        cstr = np.array([np.nan] * (len(self.A) + 1))
        temp = np.hstack((np.transpose([cstr]), table))

        print("   |\t", end = "")

        if method == 'bland':
            for i in range(0, len(self.c)):
                print("x_" + str(i + 1), end = "\t")
            for i in range(0, (len(temp[0]) - len(self.c) - 2)):
                print("w_" + str(i + 1), end = "\t")
        else:
            for i in range(0, len(self.c) + 1):
                print("x_" + str(i), end = "\t")
            for i in range(0, (len(temp[0]) - len(self.c) - 3)):
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

    ## danh cho streamlit
    def stTable(self, table, method = 'bland'):
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
        if method == 'bland':
            for i in range(0, len(self.c)):
                columns.append(f'x_{i+1}')
            for i in range(0, (len(temp[0]) - len(self.c) - 2)):
                columns.append(f'w_{i+1}')
        else:
            for i in range(0, len(self.c)):
                columns.append(f'x_{i}')
            for i in range(0, (len(temp[0]) - len(self.c) - 2)):
                columns.append(f'w_{i+1}')
        columns.append('=')
        #df=pd.DataFrame(data, columns=columns)
        st.write(pd.DataFrame(data, columns=columns))
        #st.markdown(df.to_html(index=False), unsafe_allow_html=True)
        
    def processingInput(self, A1, B):
        # 1. Kiểm tra hướng của hàm mục tiêu
        A = A1.copy()
        A = A.fillna(0)
        optimize_direction = ''
        
        if A.iloc[0,-1] == 'max':
            optimize_direction = 'max'
        else:
            optimize_direction = 'min'
        A.iloc[0,-1] = 0

        # đưa cột cuối cùng của A về dạng numerical
        A.iloc[:,-1] = pd.to_numeric(A.iloc[:,-1])
        #đưa hàng đâuf tiên của A về dạng numerical
        A.iloc[0, :] = pd.to_numeric(A.iloc[0, :])
        
        # nếu optimize_direction là max thì đưa hàm mục tiêu f(x) thành -f(-x)
        if optimize_direction == 'max':
            A.iloc[0,:] = pd.to_numeric(A.iloc[0,:])
            A.iloc[0,:] = -A.iloc[0,:]

        # 2. Xử lý ràng buộc về đẳng thức và bất đẳng thức
        # norm_arr là ma trận mới sau khi chuẩn hoá ma trận A. Dùng để làm biến đầu vào trong hàm xoay Bland
        A.iloc[0] = A.iloc[0].fillna(0)
        norm_arr = A.drop(columns=A.columns[A.shape[1] -1 -1]).to_numpy()
        nrow, ncol = norm_arr.shape
        j = 1
        for i in range(1,nrow):

            if A.iloc[i, -2] == ">=":
                norm_arr[i] = -norm_arr[i]
            elif A.iloc[i, -2] == "=":
                norm_arr = np.insert(norm_arr,j+1,[-norm_arr[j,:]],axis=0)
                j += 1
            j += 1

        # 3. Xử lý ràng buộc về dấu của biến
        # rsby là hàm result by vars: dùng để kết luận biến cũ theo biến mới sau khi chạy xong thuật toán xoay Bland
        rsby = np.eye(B.shape[1])
        k = 0
        for j in range(B.shape[1]):
            if B.iloc[0, j] == "<=":
                norm_arr[:, k] = -norm_arr[:,k]
                rsby[:, k] = -rsby[:,k]
            elif B.iloc[0, j] == ">=":
                "nothing"
            else:
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

    def printVariable(self, idx, method = 'bland'):
        if method != 'two_phase':
            if idx < len(self.c):
                return "x_" + str(idx + 1)
            else:
                return "w_" + str(idx - len(self.c) + 1)
        else:
            if idx <= len(self.c + 1):
                return "x_" + str(idx)
            else:
                return "w_" + str(idx - len(self.c))
        
    def printStepByStep(self, df_result, method = 'bland'):
        st.latex(r'\text{--- Các bước thực hiện ---}')
        st.write('Từ vựng xuất phát:\n')

        for result in df_result:
            self.stTable(result)
            in_out_ind = self.selectInOut(result, method)

            if (in_out_ind == "TVTU"):
                st.write("Từ vựng tối ưu -> dừng thuật toán.")
                break

            if (in_out_ind == "KGN" or in_out_ind == 'VSN'):
                break

            st.write(f'\nVị trí xoay: ({in_out_ind[0]}, {in_out_ind[1]})')
            st.write(f"Biến vào: ${self.printVariable(in_out_ind[1])}$")
            for i in range(0, len(result[0]) - 1):
                if (np.sum(result[0:, i]) == 1 and (-1 in result[0:, i].tolist())):
                    if result[0:, i].tolist().index(-1) == in_out_ind[0]:
                        st.write(f"Biến ra: ${self.printVariable(i)}$")
                        break
            st.write("Từ vựng mới sau khi xoay:\n")

    def printPhaseOne(self, df_result, in_out_ind):
        st.latex(r'\text{--- Các bước thực hiện ---}')
        st.write('Từ vựng xuất phát của bài toán bổ trợ:\n')

        cur_arr = df_result[0]
        self.stTable(cur_arr, 'two_phase')
        st.write(f'\nVị trí xoay: ({in_out_ind[0]}, {in_out_ind[1]})')
        st.write(f"Biến vào: ${self.printVariable(in_out_ind[1], 'two_phase')}$")
        for i in range(0, len(cur_arr[0]) - 1):
            if (np.sum(cur_arr[0:, i]) == -1 and (-1 in cur_arr[0:, i].tolist())):
                if cur_arr[0:, i].tolist().index(-1) == in_out_ind[0]:
                    st.write(f"Biến ra: ${self.printVariable(i, 'two_phase')}$")
                    break
        st.write("Từ vựng mới sau khi xoay:\n")

        for result in df_result[1:]:
            self.stTable(result, 'two_phase')
            in_out_ind = self.selectInOut(result, 'simplex')

            if (in_out_ind == "TVTU"):
                st.write("\nTừ vựng tối ưu -> dừng thuật toán.\n")
                break
            elif (in_out_ind == 'VSN'):
                break

            st.write(f'\nVị trí xoay: ({in_out_ind[0]}, {in_out_ind[1]})')
            st.write(f"Biến vào: ${self.printVariable(in_out_ind[1], 'two_phase')}$")

            for i in range(0, len(result[0]) - 1):
                if (np.sum(cur_arr[0:, i]) == -1 and (-1 in cur_arr[0:, i].tolist())):
                    if cur_arr[0:, i].tolist().index(-1) == in_out_ind[0]:
                        st.write(f"Biến ra: ${self.printVariable(i, 'two_phase')}$")
                        break

            st.write("Từ vựng mới sau khi xoay:\n")

    # phương thức xây dựng từ vựng xuất phát
    def generateTable(self):
        # xây dựng dòng đầu tiên của bảng, tương ứng hàm mục tiêu
        num_var = len(self.c)
        num_constraint = len(self.A)
        #tab1 = np.hstack((self.c, [0] * (num_constraint + 1)))
        tab1 = np.hstack((self.c, [0] * (num_constraint + 1)))
        
        A = self.A

        if (not ((num_var + num_constraint) == len(self.A[0]))):
            I = np.identity(num_constraint)
            A = np.hstack((self.A, I))
        
        tab2 = np.hstack((A, np.transpose([self.b])))
        
        table = np.vstack((tab1, tab2))
        table = np.array(table, dtype ='float')
        table[1:,:-1] = -table[1:,:-1]
        table[table == 0] = 0
        return table
    
    def findEyeList(self, arr_col):
        lst = arr_col.tolist()
        for element in lst:
            if element not in [0,-1]:
                return None
            
        if lst.count(-1) == 1:
            return lst.index(-1)

    def selectInOut(self, arr, method = 'bland'):
        """
        Output:
        - nếu i = None, tức là không có biến vào --> từ vựng tối ưu
        - nếu i != None, j = None, tức là có biến vào, không có biến ra --> bài toán không giới nội
        - nếu i != None, j != None, trả về chỉ số (i,j) cần tìm
        """

        temp = arr[0, :-1]
        # if ((np.count_nonzero(temp) < (arr.shape[1] - arr.shape[0] - 1)) and np.sum(arr[0, :]) != 1):
        #     self.type_result = 'TVTU'
        #     return('TVTU')

        ind_in = None
        ind_out = None

        if (method == 'simplex'):
            # Tìm phần tử có bi âm nhất, hoặc trả về None nếu không tồn tại phần tử âm nào.
            # min_val = np.amin(arr[0, 0:-1])
            min_val = np.amin(arr[0, 0:-1])
            if min_val < 0:
                ind_in = arr[0, 0:-1].tolist().index(min_val)
            else:
                return('TVTU')
        elif (method == 'bland'):
            # Tìm phần tử đầu tiên có bi âm, hoặc trả về None nếu không tồn tại phần tử âm nào.
            const_arr = arr[0, :-1]
            lst = const_arr[const_arr < 0]
            if len(lst) != 0:
                ind_in = np.where(const_arr < 0)[0][0]
            else:
                return('TVTU')
        
        # Tìm tỷ lệ b/|a| nhỏ nhất để chọn biến ra
        minimum = 99999999
        for i in range(1, len(arr)):
            if arr[i, ind_in] < 0:
                val = arr[i, -1] / abs(arr[i, ind_in])
                if val < minimum:
                    minimum = val
                    ind_out = i

        # Bai toan Khong gioi noi
        if ind_in != None and ind_out == None:
            return('KGN') 
        
        # Tra ve vi tri cua bien can xoay
        return(ind_out, ind_in)
    
    @staticmethod
    def selectInOutDual(tab_arr):
        """
        # input
        - tab_arr(np.array): ma trận tương tự dạng bảng của phương pháp đơn hình

        # output: luôn tồn tại i 
        - Nếu i!= None, j != None --> trả về chỉ số (j,i) cần tìm
        - Nếu i != None, j == None --> return None

        # Quy tắc chọn biến vào ra
        - Biến ra (i_0): chọn wi là biến ra có bi âm nhất 
        - Biến vào (j_0): trên dòng i_0, xét từng min{c_j/a_i0j} với (c_j > 0 và a_i0j > 0)
        """
        ind_in = None
        ind_out = None

        # Chọn biến ra là biến có bi âm nhất 
        bi = tab_arr[1:, -1]
        bi = np.array([i if i < 0 else np.inf for i in bi])
        ind_out = np.where(bi == np.min(bi))[0][0] + 1

        # Chọn biến vào là biến có {c_j/a_i0j} nhỏ nhất với (c_j > 0 và a_i0j > 0)
        row_obj_func = tab_arr[0,:-1]
        row_ind_out = tab_arr[ind_out, :-1]
        ci_ai = [row_ind_out[i]/row_obj_func[i] if (row_ind_out[i] > 0 and row_obj_func[i] > 0) else np.inf for i in range(len(row_obj_func))]
        if np.all(ci_ai != np.inf):
            ind_in = np.where(ci_ai == np.min(ci_ai))[0][0]
        else:
            # Nếu không tồn tại biến vào ind_in
            return None
        
        # Trả về chỉ số của biến vào, ra cần tìm
        return(ind_in, ind_out)
    
    @staticmethod
    def rotate(arr, rotate_point):
        nrow, ncol = arr.shape
        arr_new = arr.copy()
        arr_new[rotate_point[0]] = arr_new[rotate_point[0]]/(-arr[rotate_point])
        for row in range(nrow):
            if row != rotate_point[0]:
                for col in range(ncol):
                    if col == rotate_point[1]:
                        arr_new[row, col] = 0
                    else:
                        arr_new[row, col] = arr[row, rotate_point[1]] * arr_new[rotate_point[0], col] + arr[row, col]
        # Do lỗi lưu số thực của numpy (ví dụ: thay vì lưu số a = 0, thì có thể numpy lưu bằng -5.555557e-17) nên nhóm em chọn cách xử lý là làm tròn tới số thập phân thứ 13
        #arr_new = np.round(arr_new, decimals = 13)
        arr_new[arr_new == 0] = 0 

        return arr_new

    # Phương pháp đơn hình
    def Simplex(self, table):
        # tạo danh sách df_result để lưu lại các từ vựng tại mỗi lần xoay
        df_result = list()

        # thêm từ vựng ban đầu vào danh sách
        df_result.append(table)
        cur_arr = table.copy()

        # Xoay đơn hình cho đến khi tìm được từ vựng tối ưu
        in_out_ind = self.selectInOut(cur_arr, 'simplex')
        while (in_out_ind != 'TVTU' and in_out_ind != 'VSN'):
            if in_out_ind != "KGN":
                new_arr = self.rotate(cur_arr, in_out_ind)
                df_result.append(new_arr.copy())
            else:
                self.type_result = "KGN"
                break
            cur_arr = new_arr
            in_out_ind = self.selectInOut(cur_arr, 'simplex')

        self.type_result = self.selectInOut(df_result[-1], 'simplex')

        return df_result

    # Phương pháp Bland
    def Bland(self, table):
        # tạo danh sách df_result để lưu lại các từ vựng tại mỗi lần xoay
        df_result = list()

        # thêm từ vựng ban đầu vào danh sách
        df_result.append(table)
        cur_arr = table.copy()

        # Xoay bland cho đến khi tìm được từ vựng tối ưu
        in_out_ind = self.selectInOut(cur_arr, 'bland')
        while (in_out_ind != 'TVTU' and in_out_ind != 'VSN'):
            if in_out_ind != "KGN":
                new_arr = self.rotate(cur_arr, in_out_ind)
                df_result.append(new_arr.copy())
            else:
                self.type_result = "KGN"
                break
            cur_arr = new_arr
            in_out_ind = self.selectInOut(cur_arr, 'bland')
        
        if self.type_result == None:
            self.type_result = self.selectInOut(df_result[-1], 'bland')

        return df_result
    
    # Phương pháp hai pha
    def TwoPhase(self, tab_arr, step_by_step):
        """ 
        # input
        - norm_arr (np.array): ma trận từ vựng được tạo thành bởi hàm mục tiêu; các ràng buộc đẳng thức, bất đẳng thức;
            các ràng buộc về dấu của biến đã được chuẩn hoá
        - tab_arr (np.array): ma trận norm_arr sau khi được đưa về dạng bảng của phương pháp đơn hình

        # output
        - df_result (list): một danh sách lưu các từ vựng/ các bước giải của thuật toán 2 pha
        - type_result: phân loại kết quả của từ vựng cuối cùng trong df_result: KGN, VN, TVTU(DDN, VSN) 
        """
        if step_by_step == True:
            st.write('\nPha 1:\n')
        # 1. Pha 1
        """ 
        1.1 Lập bài toán bổ trợ  
        - Thêm biến x0 vào ràng buộc đẳng thức, bất đẳng thức. Lập từ vựng xuất phát C = x0  
        - Đưa về dạng ma trận để có thể giải, lưu vào biến tab_C1 (np.array)
        """
        tab_C1 = tab_arr.copy()
        tab_C1 = np.insert(tab_C1, 0, np.ones(tab_C1.shape[0]), axis = 1)
        tab_C1[0, 1:] = 0
        tab_C1[0, 0] = 1

        """ 
        1.2 Xoay từ vựng 
        - Từ vựng đầu tiên
        + Chọn biến vào: x0
        + Chọn biến ra: biến ở dòng ứng với bi âm nhất, ưu tiên dòng có hệ số nhỏ nhất  
        - Từ vựng thứ 2 trở đi: áp dụng cách xoay đơn hình --> từ vựng cuối pha 1
        """
        # chọn biến vào là x_0
        j = 0
        # chọn biến ra, do ở thuật toán 2 pha, luôn tồn tại bi<0 nên luôn tồn tại biến ra i
        bi = tab_C1[1:, -1]
        i = np.where(bi == np.min(bi[bi < 0]))[0][0] + 1

        # tạo df_result để lưu các từ vựng 
        df_C1 = list()
        # xoay từ vựng
        cur_arr = tab_C1
        df_C1.append(cur_arr.copy())
        rotate_point = (i, j)

        in_out_ind = (i,j)
        while (in_out_ind != 'TVTU' and in_out_ind != 'VSN'):
            if in_out_ind != "KGN":
                new_arr = self.rotate(cur_arr, in_out_ind)
                df_C1.append(new_arr.copy())
            else:
                self.type_result = "KGN"
                break
            cur_arr = new_arr
            in_out_ind = self.selectInOut(cur_arr, 'simplex')

        if step_by_step == True:
            self.printPhaseOne(df_C1, rotate_point)
            if (self.type_result == 'VSN'):
                return(df_C1)

        """ 
        1.3 Kiểm tra từ vựng cuối pha 1 (từ vựng tối ưu của P1), xét hàm mục tiêu
        - Nếu hàm mục tiêu chỉ có biến x0 --> chuyển sang pha 2
        - Nếu hàm mục tiêu gồm biến x0 và những biến khác (hoặc không có x0) --> kết luận bài toán vô nghiệm --> dừng thuật toán. 
        """
        # hàm mục tiêu của từ vựng cuối pha 1
        obj_func_last_C1 = df_C1[-1][0,:-1]
        # Nếu hàm mục tiêu chỉ có biến x0 --> pha 2
        if obj_func_last_C1[0] == 1 and np.all(obj_func_last_C1[1:] == 0):
            if step_by_step == True:
                st.write('Pha 2:')
            # 2. Pha 2
            """ 
            2.1 Tạo từ vựng mới
            - Từ từ vựng tối ưu của pha 1 (P1) -> cho x_0 = 0, lấy các ràng buộc ở từ vựng này
            - Dùng hàm mục tiêu gốc (P) kết hợp với hàm mục tiêu (P1) -> hàm mục tiêu mới
            """
            # Xoá cột x_0
            temp_arr_last_C1 = df_C1[-1].copy()
            temp_arr_last_C1 = np.delete(temp_arr_last_C1,0,1)
            # tìm số biến, số ràng buộc của bài toán gốc (P)
            n_const = temp_arr_last_C1.shape[0] - 1
            n_var = temp_arr_last_C1.shape[1]-temp_arr_last_C1.shape[0]
            # tạo một ma trận biểu diễn x1, x2,... từ hàm mục tiêu ban đầu

            obj_fun_z = self.c.reshape(1,-1)
            # tạo một ma trận biểu diễn x1, x2,... từ hàm mục tiêu theo x1, x2, ..., w1, w2, ... có được trong từ vựng cuối pha 1,
            #       với shape = temp_arr_last_C1.shape
            rsby_C1 = np.zeros((n_var, n_const + n_var + 1))

            for i in range(n_var):
                if self.findEyeList(temp_arr_last_C1[:,i]) != None:
                    j = self.findEyeList(temp_arr_last_C1[:,i])
                    rsby_C1[i] = temp_arr_last_C1[j]
                    rsby_C1[i,i] = 0
                else:
                    rsby_C1[i,i] = 1

            self.z = obj_fun_z@rsby_C1
            # kết hợp z với ràng buộc của từ vựng cuối pha 1 --> tạo thành từ vựng đầu pha 2 tab_C2
            tab_C2 = np.insert(self.z, 1, temp_arr_last_C1[1:], axis = 0)

            """ 
            2.2 Tiến hành xoay Bland trên từ vựng mới --> kết luận nghiệm
            """
            df_C2 = list()
            cur_arr = tab_C2

            df_C2.append(cur_arr.copy())
            # Thuc hien don hinh cho đến khi tìm được từ vựng tối ưu
            in_out_ind = self.selectInOut(cur_arr, 'simplex')
            while (in_out_ind != 'TVTU' and in_out_ind != 'VSN'):
                if in_out_ind != "KGN":
                    new_arr = self.rotate(cur_arr, in_out_ind)
                    df_C2.append(new_arr.copy())
                else:
                    self.type_result = "KGN"
                    break
                cur_arr = new_arr
                in_out_ind = self.selectInOut(cur_arr, 'simplex')

            self.type_result = self.selectInOut(df_C2[-1], 'simplex')

            return df_C2

        # Nếu hàm mục tiêu gồm biến x0 và những biến khác (hoặc không có x0) --> kết luận bài toán vô nghiệm --> dừng thuật toán. 
        else:
            self.type_result = "VN"
            # self.call_result(df_C1, step_by_step)
            return df_C1

    # Phương pháp đơn hình đối ngẫu
    def DualSimplex(self, tab_arr):
        """ 
        # input
        - tab_arr (np.array): ma trận từ vựng dạng bảng của phương pháp đơn hình

        # output
        - df_result (list): một danh sách lưu các từ vựng/ các bước giải của thuật toán 2 pha
        - type_result: phân loại kết quả của từ vựng cuối cùng trong df_result: KGN, VN, TVTU(DDN, VSN) 

        """

        # tạo danh sách df_result để lưu lại các từ vựng tại mỗi lần xoay
        df_result = list()
        # thêm từ vựng ban đầu vào danh sách
        df_result.append(tab_arr)
        cur_arr = tab_arr.copy()

        bi = cur_arr[1:, -1]
        # Nếu vẫn còn tồn tại bi < 0, chọn biến vào ra bằng hàm select_in_out_ind_selfdual --> xoay từ vựng
        while np.any(bi < 0):
            if self.selectInOutDual(cur_arr) != None:
                in_out_ind = self.selectInOutDual(cur_arr)
                new_arr = self.rotate(cur_arr, in_out_ind)
                df_result.append(new_arr)
                cur_arr = new_arr
                bi = cur_arr[1:, -1]
            else:
                # Do phần này thầy chưa dạy, nên đặt tạm type_result = "VN"
                self.type_result = "VN"
                return(df_result, self.type_result)

        # Nếu bi >= 0, chọn biến vào ra bằng hàm select_input_output_index --> xoay từ vựng như phương pháp đơn hình/bland
        in_out_ind = self.selectInOut(cur_arr)
        while (in_out_ind != 'TVTU' and in_out_ind != 'VSN'):
            if in_out_ind != "KGN":
                new_arr = self.rotate(cur_arr, in_out_ind)
                df_result.append(new_arr)
            else:
                self.type_result = "KGN"
                break
            cur_arr = new_arr
            in_out_ind = self.selectInOut(cur_arr)
        
        self.type_result = self.selectInOut(df_result[-1])

        return df_result

    @staticmethod
    def to_self_dual(A, B):
        """ 
        # goal
        - Tìm bài toán đối ngẫu (D) của bài toán quy hoạch tuyến tính ban đầu (P)

        # input
        - A (dataframe): lưu các ràng buộc đẳng thức, bất đẳng thức của bài toán (P)
        - B (dataframe): lưu các ràng buộc về dấu của biến của bài toán (P)

        # output 
        - new_const (dataframe): lưu các ràng buộc đẳng thức, bất đẳng thức của bài toán (D)
        - new_vars (dataframe): lưu các ràng buộc về dấu của biến của bài toán (D)
        """

        A = A.copy()
        B = B.copy()
        A = A.fillna(0)
        # array lưu các ràng buộc về đẳng thức, bất đẳng thức của bài toán mới
        new_const = A.to_numpy().transpose()
        new_const[-1,1:] = pd.to_numeric(new_const[-1,1:]) 
        new_const = np.concatenate((new_const[-1,:].reshape(1,-1),new_const[:-1,:]), axis=0)
        new_const = np.concatenate((new_const[:,1:], new_const[:,0].reshape(-1,1)), axis=1)

        # array lưu các ràng buộc về dấu của biến trong bài toán mới
        new_vars_sign = new_const[-1,:-1]
        new_const = new_const[:-1]
        new_const_sign = B.to_numpy(dtype=object)[0]
        
        # set optimize_direction cho bài toán mới
        optimize_direction_old = A.iloc[0,-1]
        if optimize_direction_old == "min":
            new_const[0,-1] = "max"
        else:
            new_const[0,-1] = "min"

        # set lại dấu cho ràng buộc đẳng thức, bất đẳng thức trong bài toán mới
        if optimize_direction_old == "min":
            # chuyển dấu của biến -> dấu của ràng buộc
            for i in range(len(new_const_sign)):
                if new_const_sign[i] == ">=":
                    new_const_sign[i] = "<="
                elif new_const_sign[i] == "<=":
                    new_const_sign[i] = ">="
                else:
                    new_const_sign[i] = "="

            # chuyển dấu của ràng buộc thành dấu của biến
            for i in range(len(new_vars_sign)):
                if new_vars_sign[i] == ">=":
                    new_vars_sign[i] = ">="
                elif new_vars_sign[i] == "<=":
                    new_vars_sign[i] = "<="
                else:
                    new_vars_sign[i] = 0
        else:
            # chuyển dấu của biến -> dấu của ràng buộc
            for i in range(len(new_const_sign)):
                if new_const_sign[i] == ">=":
                    new_const_sign[i] = ">="
                elif new_const_sign[i] == "<=":
                    new_const_sign[i] = "<="
                else:
                    # new_const_sign[i] = str(new_const_sign[i])
                    new_const_sign[i] = "="

            # chuyển dấu của ràng buộc thành dấu của biến
            for i in range(len(new_vars_sign)):
                if new_vars_sign[i] == ">=":
                    new_vars_sign[i] = "<="
                elif new_vars_sign[i] == "<=":
                    new_vars_sign[i] = ">="
                else:
                    new_vars_sign[i] = 0

        new_const_sign = np.insert(new_const_sign, 0, 0)
        new_const = np.insert(new_const, -1, new_const_sign.reshape(1,-1), axis = 1)
        new_const = pd.DataFrame(np.array(new_const), columns=None)
        new_vars = pd.DataFrame(np.array(new_vars_sign.reshape(1,-1)),columns=None)

        return(new_const, new_vars)
    
    def backward_primal_tab(self, dual_arr):
        """ 
        # goal 
        Đưa từ vựng của bài toán đối ngẫu (D) về từ vựng của bài toán (P)

        # Input
        - dual_arr : ma trận từ vựng tối ưu của bài toán (D)

        # Output
        - primal_arr: ma trận từ vựng tối ưu của bài toán (P)

        """
        # Tính số lượng biến, số lượng ràng buộc trong mỗi bài toán
        nvar_dual = dual_arr.shape[1] - dual_arr.shape[0]
        nconst_dual = dual_arr.shape[0] - 1 

        nvar_primal = nconst_dual
        nconst_primal = nvar_dual

        # Sắp xếp lại dual_arr theo thứ tự của primal_arr
        tmp_arr = np.concatenate([dual_arr[:, nvar_dual:-1], dual_arr[:, : nvar_dual], dual_arr[:,-1].reshape(-1,1)], axis=1)

        eye_list_j = [i for i in range(tmp_arr.shape[1] - 1) if self.findEyeList(tmp_arr[:, i])!= None]
        tmp_arr_shortcut = np.delete(tmp_arr, eye_list_j, axis = 1)
        tmp_arr_shortcut = np.concatenate([tmp_arr_shortcut[:,-1].reshape(-1,1), tmp_arr_shortcut[:,:-1]], axis = 1).T
        
        # Tạo ma trận primal_arr
        primal_arr = np.zeros((nconst_primal+1, nconst_primal + nvar_primal + 1))
        primal_arr[:,-1] = tmp_arr_shortcut[:,0]
        eye_list_i = [self.findEyeList(tmp_arr[:, j]) for j in eye_list_j]
        for n in range(len(eye_list_j)):
            i = eye_list_i[n]
            j = eye_list_j[n]
            primal_arr[:,j] = tmp_arr_shortcut[:,i]
            primal_arr[1:,j] = -primal_arr[1:,j]
        
        j = 1
        for i in range(primal_arr.shape[1] - 1):
            if np.all(primal_arr[:,i] == 0):
                primal_arr[j,i] = -1
                j += 1

        return(primal_arr)

    def call_result(self, df_result, step_by_step = True, method = 'bland'):
        """ 
        # input
        - rsby: ma trận biểu diễn của 1 biến
        - optimize_direction: biến lưu hướng tối ưu của hàm mục tiêu (max or min)
        - df_result: list lưu các bước giải của bài toán QHTT
        - type_result: phân loại kết quả của từ vựng cuối cùng trong df_result: KGN, VN, TVTU(DDN, VSN)

        # output
        - show các bước giải, nếu lựa chọn show
        - show kết quả cuối cùng: 
            + btoan có nghiêm duy nhất: show nghiêm tối ưu, giá trị tối ưu
            + btoan vô nghiệm: nếu optimize_direction là max thì giá trị tối ưu là -inf, ngược lại
            + btoan không giới nội: nếu optimize_direction là max thì giá trị tối ưu là inf, ngược lại
            + btoan vô số nghiệm: cho các biến không cơ sở xuất hiện ở hàm mục tiêu bằng 0, viết biến cơ sở theo điều kiện các biến không cơ sở
        """

        arr_tvtu = df_result[-1] # ma trận từ vựng tối ưu
        tvtu = arr_tvtu[0, :-1] # từ vựng tối ưu không lấy bi

        if method == 'dual':
            if step_by_step == True:
                self.printStepByStep(df_result[0:-1], 'simplex')

            #st.write('\nTừ vựng tối ưu tương ứng của bài toán gốc: \n')
            nvar_dual = df_result[-2].shape[1] - df_result[-2].shape[0]
            nconst_dual = df_result[-2].shape[0] - 1 
            nvar_primal = nconst_dual
            nconst_primal = nvar_dual
            self.setA(arr_tvtu[1:, :-1])
            self.setB(arr_tvtu[1:,-1])
            self.setC(arr_tvtu[0, 0:nvar_primal])
            if step_by_step == True:
                st.write('\nTừ vựng tối ưu tương ứng của bài toán gốc: \n')
                self.stTable(arr_tvtu)

        # tách type_result "TVTU" thành "DNN" và "VSN"
        if self.type_result == "TVTU":
            # Tính số biến tự do
            nvar_free = self.rsby.shape[1] - self.rsby.shape[0] 
            # Tính số biến bị khuyết ở hàm mục tiêu so với ràng buộc
            n_missing = 0
            n_const = arr_tvtu.shape[0] - 1 
            for i in range(len(tvtu)):
                if tvtu[i] == 0 and np.any(arr_tvtu[1:,i] != 0):
                    n_missing += 1

            n_missing = n_missing - n_const - nvar_free
            
            if n_missing >= 1:
                self.type_result = "VSN"
                # st.write(n_missing)
            else:
                # in ra các bước làm nếu step_by_step = True
                if (step_by_step == True and method != 'dual'):
                    self.printStepByStep(df_result, method)
                self.type_result = "DNN"

        # Xét các trường hợp 
        """ 
        - x_P_hat là ma trận chứa nghiệm của các biến x1, ..., xn của (P')
        - x_P là ma trận chứa nghiệm của các biến x1, ..., xn của (P) aka kết quả x cuối cùng
        - z là giá trị tối ưu của bài toán
        """

        ## in ra các bước làm nếu step_by_step = True
        # if (step_by_step == True):
        #     self.printStepByStep(df_result, method)

        x_P_hat = None
        # 1. Duy nhất nghiệm
        if self.type_result == "DNN":
            arr_tvtu = df_result[-1]
            result = np.zeros(arr_tvtu.shape[1])
            result[-1] = arr_tvtu[0, -1]
            for col in range(df_result[-1].shape[1] - 1):
                ind = self.findEyeList(arr_tvtu[:,col])
                if ind != None:
                    result[col] = arr_tvtu[ind, -1]
                else:
                    result[col] = 0
            n_const = arr_tvtu.shape[0] - 1
            x_P_hat = result[:(len(result) - n_const - 1)]
            if method != 'dual':
                self.x = self.rsby @ np.array(x_P_hat)
            else:
                self.x = x_P_hat

            if self.optimize_direction == 'min':
                self.z = result[-1]
            else:
                self.z = -result[-1]
        
        # 2. Không giới nội
        elif self.type_result == "KGN":
            if df_result != None and step_by_step == True:
                self.printStepByStep(df_result, 'simplex')
            if self.optimize_direction == 'min':
                self.z = -np.inf
            else:
                self.z = np.inf

        # 3. Vô nghiệm
        elif self.type_result == "VN":
            if df_result != None and step_by_step == True:
                self.printStepByStep(df_result, 'simplex')
            if self.optimize_direction == 'min':
                self.z = np.inf
            else:
                self.z = -np.inf

        elif self.type_result == "VSN":
            """
            Xét trên từ vựng tối ưu ci>=0. Nếu tồn tại ci = 0 --> hàm mục tiêu bị khuyết so với các ràng buộc --> VSN
            - Cho biến cơ sở ở hàm mục tiêu bằng 0
            - Viết biến cơ sở theo biến khuyết
            - Tìm điều kiện, hệ điều kiện cho biến khuyết
            + Chỉ có 1 biến khuyết: tìm khoảng giới hạn của biến khuyết (gộp chung với có 2 biến khuyết)
            + Có từ 2 biến khuyết trở lên: viết hệ điều kiện 
            """
            # x_P lúc này dùng để lưu điều kiện cho biến x1, x2,..., w1, w2, ... được lưu với cú pháp 0:">=", 1:"="
            # z để lưu hệ điều kiện cho biến dưới dạng ma trận dòng x1, x2, ..., wn; cột x1, x2, ..., wn, bi
            if method != 'dual' and step_by_step == True:
                self.printStepByStep(df_result, 'simplex')
                
            x_P = tvtu
            self.z = arr_tvtu[0,-1]
            tmp_arr_tvtu = arr_tvtu.copy()
            for i in range(len(x_P)):
                if x_P[i] != 0:
                    x_P[i] = 1
                    tmp_arr_tvtu[:, i] = 0
            cond_xP = np.zeros((tmp_arr_tvtu.shape[1] - 1, tmp_arr_tvtu.shape[1]))
            for i in range(cond_xP.shape[0]):
                j = self.findEyeList(tmp_arr_tvtu[:,i])
                if j != None:
                    cond_xP[i] = tmp_arr_tvtu[j]
                    cond_xP[i,i] = 0
            self.x = [x_P, cond_xP]
            if self.optimize_direction == max:
                self.z = -self.z
            
        self.printSolution(step_by_step)

    def solving_LP_problem(self, A, B, step_by_step = True):
        """
        # goal
        Hàm giải quyết bài toán quy hoạch tuyến tính (solving linear programming problem) bằng ba cách
        - Sử dụng thuật toán Bland
        - Sử dụng thuật toán đơn hình hai pha
        - Sử dụng thuật toán đơn hình đối ngẫu

        # input 
        - A (dataframe): lưu trữ các ràng buộc đẳng thức, bất đẳng thức của bài toán
        - B (dataframe): lưu trữ các ràng buộc về dấu của biến trong bài toán

        # output
        - Nếu bài toán được cho nằm trong 3 cách giải ứng với ba thuật toán trên thì in ra kết quả x, z và các bước giải
        - Ngược lại, trả về rỗng
        """

        nconst_A = A.shape[0] - 1
        self.processingInput(A, B)

        tab_arr = self.generateTable()
        method = None

        # Nếu bài toán có 2 ràng buộc và hoặc là tất cả ci >= 0 hoặc tất cả ci <= 0
        if (nconst_A == 2) and (np.all(self.c >= 0) or np.all(self.c <= 0)):
            # Tạo bài toán đối ngẫu (P) của A, B
            C, D = LinearProgramming.to_self_dual(A, B)
            self.processingInput(C, D)

            st.write('Đưa về bài toán đối ngẫu:\n')
            st.write("A =", self.A)
            st.write("b =", self.b)
            st.write("c =", self.c)

            tab_arr1 = self.generateTable()

            # Nếu bài toán (P) không nằm trong dạng các thuật giải đã học --> return
            if np.all(self.c >= 0) and np.all(self.b >= 0):
                df_result = [tab_arr]
                self.type_result = 'TVTU'
                self.processingInput(A, B)
                self.call_result(df_result, step_by_step, 'simplex')
                return
            
            # Nếu tồn tại ci < 0, với mọi bi >= 0 --> thuật toán Bland
            if np.any(self.c < 0) and np.all(self.b >= 0):
                st.write('\nSử dụng phương pháp xoay Bland')
                df_result = self.Bland(tab_arr1)
            # Nếu tồn tại bi < 0 --> thuật toán đơn hình đối ngẫu hoặc thuật toán hai pha
            elif np.any(self.b < 0):
                st.write('\nSử dụng thuật toán 2 pha')
                df_result = self.TwoPhase(tab_arr1, step_by_step)
            
            # Nếu (D) không giới nội thì (P) vô ngiệm, ngược lại, nếu (D) vô nghiệm thì (P) không giới nội
            # Nếu bài toán nằm trong 2 trường hợp này thì không cần chuyển từ vựng tối ưu của (D) thành từ vựng tối ưu của(P)
            if self.type_result == 'KGN':
                self.type_result = 'VN'
            elif self.type_result == 'VN':
                self.type_result = 'KGN'
            else:
                # Chuyển ma trận từ vựng tối ưu của (D) thành ma trận từ vựng tối ưu của (P)
                primal_arr = self.backward_primal_tab(df_result[-1])
                df_result.append(primal_arr)
                method = 'dual'

            self.call_result(df_result, step_by_step, method)
            return
        if step_by_step == True:
            st.write('Dạng chuẩn A x c = b:\n')
            st.write("A =", self.A)
            st.write("b =", self.b)
            st.write("c =", self.c)

        # Nếu tồn tại ci < 0, với mọi bi >= 0 --> thuật toán Bland
        if np.any(self.c < 0) and np.all(self.b >= 0):
            st.write('\nSử dụng phương pháp xoay Bland')
            df_result = self.Bland(tab_arr)
            method = 'bland'
        # Nếu tồn tại bi < 0 --> thuật toán đơn hình đối ngẫu hoặc thuật toán đơn hình hai pha
        if np.any(self.b < 0):
            st.write('Sử dụng thuật toán 2 pha')
            df_result = self.TwoPhase(tab_arr, step_by_step)
            method = 'simplex'

        # Trả kết quả của x, z và các bước giải quyết bài toán
        self.call_result(df_result, step_by_step, method)