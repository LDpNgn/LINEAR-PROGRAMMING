# THUẬT TOÁN GIẢI QUYẾT BÀI TOÁN QUY HOẠCH TUYẾN TÍNH TỔNG QUÁT

## 1. Xử lý đầu vào
1.1 Về ràng buộc đẳng thức, bất đẳng thức --> chuyển về dạng chuẩn  
- nếu <=: giữ nguyên  
- nếu >=: nhân với -1  
- nếu =: chuyển ràng buộc Ax = b thành 2 ràng buộc Ax <= b và -Ax <= -b

1.2 Về ràng buộc dấu của biến 
- x>=0: giữ nguyên
- x<=0: thay x=-x, ghi nhớ vị trí biến để kết luận x=-x
- x không có ràng buộc: tạo 2 biến +x, -x mới, ghi nhớ vị trí biến để kết luận x = (+x) - (-x)

1.3 Về hướng của hàm mục tiêu  
- Nếu optimize_direction là max thì đưa hàm mục tiêu f(x) thành -f(-x)
- Nếu optimize_direction là min thì giữ nguyên

## 2. Thuật toán đơn hình Danzitg
Thuật toán đơn hình Bland để giải quyết bài toán QHTT khi:
- Tồn tại ci < 0 trên hàm mục tiêu
- Với mọi bi > 0 trên ràng buộc đẳng thức, bất đẳng thức

Thuật toán
- Chọn biến vào ra
+ Chọn biến vào: biến có hệ số âm nhất trên hàm mục tiêu
+ Chọn biến ra: 

...
## 3. Thuật toán đơn hình Bland
Thuật toán đơn hình Bland để giải quyết bài toán QHTT khi:
- Tồn tại ci < 0 trên hàm mục tiêu
- Với mọi bi >= 0 trên ràng buộc đẳng thức, bất đẳng thức

... ghi thuật toán ...

## 4. Thuật toán đơn hình hai pha
Thuật toán đơn hình hai pha để giải quyết bài toán QHTT khi:
- Tồn tại ci < 0 trên hàm mục tiêu
- Tồn tại bi < 0 trên ràng buộc đẳng thức, bất đẳng thức

Thuật toán:
3.1 Pha 1 
- Lập bài toán bổ trợ
+ Thêm x_0 vào từ vựng ban đầu 
+ Lập từ vựng xuất phát (C) với hàm mục tiêu chỉ gồm biến x_0
- Chọn biến vào ra
+ Chọn biến vào: x_0
+ Chọn biến ra: biến ứng với dòng bi âm nhất, ưu tiên dòng có hệ số nhỏ nhất
- Xoay từ vựng như đơn hình
- Từ từ vựng thứ 2, áp dụng cách xoay như đơn hình để tìm được từ vựng cuối pha 1
- Kiểm tra từ vựng cuối pha 1
+ Nếu hàm mục tiêu chỉ gồm biến x0 --> chuyển qua pha 2
+ Ngược lại --> kết luận bài toán vô nghiệm

3.2 Pha 2
- Từ từ vựng tối ưu của pha 1
+ Cho x_0 = 0, lấy các ràng buộc ở từ vựng này (*)
+ Dùng hàm mục tiêu gốc (P') kết hợp với ràng buộc (*) --> hàm mục tiêu mới
- Xây dựng từ vựng mới bằng hàm mục tiêu mới kết hợp với các ràng buộc khi cho x_0 = 0
- Tiến hành xoay đơn hình trên từ vựng mới
- Kết luận nghiệm

## 5. Thuật toán đơn hình đối ngẫu
Thuật toán đơn hình đối ngẫu để giải quyết bài toán QHTT khi: 
- Tồn tại bi < 0 trên ràng buộc đẳng thức, bất đẳng thức

4.1 Bước 1: Xây dựng từ vựng xuất phát như bài toán đơn hình 

4.2 Bước 2: Quy tắc chọn biến vào ra 
- Nếu vẫn còn tồn tại bi < 0
+ Biến ra (i_0): chọn wi là biến ra có bi âm nhất 
+ Biến vào (j_0): trên dòng i_0, xét từng min{c_j/a_i0j} với(c_j > 0 và a_i0j > 0)
+ Xoay
- Nếu bi >= 0: giải quyết bài toán theo phương pháp đơn hình gốc/Bland

- Nếu {c_j/a_i0j} với(c_j > 0 và a_i0j > 0) không tồn tại --> Phương pháp khác

## 6. Dùng kỹ thuật đối ngẫu để giải bài toán
- Bước 1: Tìm bài toán đối ngẫu (D) của bài toán (P)
- Bước 2: Giải bài toán (D) bằng thuật toán đơn hình Bland hoặc đơn hình hai pha
- Bước 3: Chuyển từ vựng tối ưu của bài toán (D) sang từ vựng tối ưu của bài toán (P)
- Bước 4: Kết luận nghiệm theo bài toán (P) ban đầu

