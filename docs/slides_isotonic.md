# Slide: Isotonic Regression Binning
## 10 trang — dùng cho Marp / reveal.js hoặc chuyển sang PDF

---

## Slide 1 — Tiêu đề

# Isotonic Regression Binning

**Binning đơn điệu dựa trên hồi quy đẳng áp**

- Ứng dụng: Scorecard, risk modeling, credit scoring  
- Đảm bảo: Biến bin có quan hệ **đơn điệu** (monotonic) với event rate  
- Công cụ: Isotonic Regression (sklearn) + quantile chia sơ bộ  

---

## Slide 2 — Binning là gì?

# Binning trong modeling

- **Binning**: Chia biến liên tục X thành một số khoảng (bins), mỗi khoảng gán một giá trị (index, WOE, score).
- **Mục đích**: 
  - Giảm nhiễu, dễ giải thích cho business  
  - Xử lý phi tuyến (X ↔ risk)  
  - Chuẩn hóa đầu vào cho scorecard (WOE, điểm)  
- **Thách thức**: Chọn **cut-points** sao cho vừa đủ bins, vừa ổn định, vừa có ý nghĩa (ví dụ đơn điệu).

---

## Slide 3 — Tại sao cần monotonic?

# Monotonicity (đơn điệu)

- **Monotonic**: Khi X tăng (hoặc giảm), event rate (bad rate / risk) chỉ đi **một chiều** — tăng dần hoặc giảm dần, không lên xuống bất thường.
- **Lý do cần**:
  - **Giải thích**: "Thu nhập càng cao → rủi ro càng thấp" dễ bán với business và compliance.  
  - **Ổn định**: Tránh bin "lồi lõm" do nhiễu mẫu.  
  - **Regulation**: Nhiều quy định (ví dụ fair lending) yêu cầu giải thích đơn điệu.
- **Isotonic Binner**: Tạo cut-points sao cho **sau khi bin, event rate theo bin luôn đơn điệu**.

---

## Slide 4 — Isotonic Regression là gì?

# Isotonic Regression (hồi quy đẳng áp)

- **Định nghĩa**: Tìm hàm \( \hat{f}(x) \) **đơn điệu** (tăng hoặc giảm) sao cho xấp xỉ dữ liệu tốt nhất (thường là tổng bình phương sai số có trọng số).
- **Ràng buộc**: Chỉ cho phép đường đi lên (increasing) hoặc đi xuống (decreasing), **không** lên xuống tùy ý.
- **Hình ảnh**: Đường **bậc thang** (step function) — các bậc ngang = các mức giá trị bằng nhau; chỗ **gãy** giữa hai bậc = ranh giới (cut-point) tiềm năng.
- **Trong binning**: Input = (center của bin, event rate của bin); output = đường bậc thang đơn điệu; **điểm gãy** → dùng làm cut-point trên trục X.

---

## Slide 5 — Ý tưởng trực quan

# Từ dữ liệu nhiễu → đường đơn điệu

```
Event rate
    ^
    |     *  *
    |   *   *    *
    | *       *     *   ← Dữ liệu thô (bin nhỏ) lên xuống do nhiễu
    |   *   *  *
    +-------------------------> X (hoặc bin center)
```

- Isotonic Regression vẽ **đường bậc thang** đi LÊN (hoặc XUỐNG) mượt nhất qua các điểm đó.  
- **Mỗi bậc ngang** = một nhóm bin có cùng event rate (đã làm mượt).  
- **Điểm gãy** giữa hai bậc = **cut-point** trên trục X → dùng để chia bin cuối cùng.

---

## Slide 6 — Thuật toán: Tổng quan 4 bước

# Các bước của Isotonic Binner

| Bước | Mô tả ngắn |
|------|------------|
| **1** | Chia X theo **quantile** (vd. 20 bins) → có `init_cuts`, `edges`, `bin_idx`. |
| **2** | Với mỗi bin: tính **center** (median X), **event rate**, **weight** (số lượng). |
| **3** | **Fit Isotonic Regression**: input (centers, rates, weights), output là đường bậc thang đơn điệu (chiều theo `direction_`). |
| **4** | **Điểm gãy**: Nơi hai bậc liên tiếp khác nhau → lấy điểm giữa hai center, ánh xạ về giá trị X thực → **cut-point**. Cuối cùng giới hạn số cut theo `max_bins`. |

---

## Slide 7 — Bước 1 & 2: Quantile + Event rate

# Chi tiết bước 1 và 2

- **Bước 1 — Quantile**  
  - Dùng `quantile_cuts(x, n_init_bins)` → danh sách cut-points.  
  - `edges = [-∞, cuts..., +∞]`, `bin_idx = pd.cut(x, edges)`.

- **Bước 2 — Đại diện mỗi bin**  
  - **Center**: `median(x[mask])` — đại diện vị trí bin trên trục X.  
  - **Rate**: `y[mask].mean()` — event rate của bin.  
  - **Weight**: `mask.sum()` — số lượng mẫu (dùng trong fit có trọng số).  

→ Ta có bộ ba **(centers, rates, weights)** làm đầu vào cho Isotonic Regression.

---

## Slide 8 — Bước 3 & 4: Fit Isotonic + Điểm gãy

# Chi tiết bước 3 và 4

- **Bước 3 — Fit Isotonic**  
  - `IsotonicRegression(increasing=(direction_=="ascending"), out_of_bounds="clip")`  
  - `fitted = iso.fit_transform(centers, rates, sample_weight=weights)`  
  - `fitted` là đường bậc thang đơn điệu (cùng chiều với risk).

- **Bước 4 — Cut-points**  
  - Duyệt cặp `(fitted[i], fitted[i+1])`: nếu khác nhau đáng kể → có **điểm gãy**.  
  - Boundary trên X: `(centers[i] + centers[i+1]) / 2` → tìm giá trị X thực gần nhất (trong `sorted_x`) → thêm vào `cuts`.  
  - Cuối: `cuts = sorted(set(cuts))`, cắt bớt nếu `len(cuts) >= max_bins` (giữ `max_bins - 1` cut).

---

## Slide 9 — Ưu điểm & Hạn chế

# Ưu điểm

- **Đơn điệu đảm bảo**: Sau khi bin, event rate theo bin luôn tăng hoặc giảm đều.  
- **Tận dụng thông tin**: Dùng toàn bộ (center, rate, weight) qua Isotonic có trọng số.  
- **Ổn định**: Làm mượt nhiễu thay vì cắt theo từng bin nhỏ lẻ.  
- **Tích hợp sẵn**: Dùng `sklearn.isotonic.IsotonicRegression`, dễ bảo trì.

# Hạn chế

- Phụ thuộc **số bin khởi tạo** (`n_init_bins`): quá ít → ít điểm gãy; quá nhiều → có thể overfit.  
- **Một chiều**: Chỉ mô hình quan hệ đơn điệu, không mô hình dạng U hay phức tạp hơn (khi đó cần method khác hoặc biến đổi X).

---

## Slide 10 — Tóm tắt

# Tóm tắt

1. **Isotonic Binner** dùng **Isotonic Regression** để tìm đường bậc thang đơn điệu qua (center, event rate) của các bin quantile.  
2. **Cut-points** = các điểm gãy của đường bậc thang, ánh xạ ngược về giá trị X.  
3. Đảm bảo **monotonic** sau khi bin, phù hợp scorecard và quy định.  
4. Luồng: **Quantile → centers/rates/weights → Isotonic fit → điểm gãy → cuts**, có giới hạn `max_bins`.

**Tài liệu tham khảo trong project**: `binning_process/supervised/isotoic.py` — class `IsotonicBinner`.
