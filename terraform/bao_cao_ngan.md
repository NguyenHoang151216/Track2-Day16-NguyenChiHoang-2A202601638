# Báo cáo kết quả benchmark LightGBM trên CPU

Mô hình LightGBM được huấn luyện trên CPU EC2 `t3.small` với thời gian load dữ liệu khoảng 2,51 giây và thời gian training chỉ 2,66 giây.  
Quá trình huấn luyện dừng tại iteration thứ 13 nhờ cơ chế early stopping, giúp hạn chế thời gian xử lý và tránh overfitting.  
Mô hình đạt AUC-ROC 0,9634, cho thấy khả năng phân biệt giao dịch gian lận và bình thường rất tốt.  
F1-Score đạt 0,80, trong khi Precision là 0,8041 và Recall là 0,7959, thể hiện sự cân bằng tương đối giữa phát hiện gian lận và hạn chế cảnh báo sai.  
Độ chính xác tổng thể đạt 99,93%, tuy nhiên cần xem xét cùng F1 và AUC vì dataset có tỷ lệ gian lận rất thấp.  
Độ trễ inference cho một giao dịch chỉ khoảng 1,22 ms, đáp ứng tốt nhu cầu dự đoán gần thời gian thực.  
Thông lượng khi dự đoán batch 1.000 dòng đạt khoảng 569.268 dòng/giây.  
Kết quả cho thấy LightGBM có thể huấn luyện nhanh và inference hiệu quả trên CPU cấu hình nhỏ mà không cần sử dụng GPU.
