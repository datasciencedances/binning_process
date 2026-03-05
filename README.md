# Monotonic WOE Binning — Tài Liệu Kỹ Thuật

> **Phiên bản:** 2.0 &nbsp;|&nbsp; **Cập nhật:** 2026 &nbsp;|&nbsp; **Package:** `binning_process`

---

## Mục lục

1. [Tổng quan & Động lực](#1-tổng-quan--động-lực)
2. [Nền tảng lý thuyết](#2-nền-tảng-lý-thuyết)
   - 2.1 [Weight of Evidence (WOE)](#21-weight-of-evidence-woe)
   - 2.2 [Information Value (IV)](#22-information-value-iv)
   - 2.3 [Ràng buộc Monotonicity](#23-ràng-buộc-monotonicity)
3. [Kiến trúc Package](#3-kiến-trúc-package)
4. [Monotonic Merge — Cơ chế chung](#4-monotonic-merge--cơ-chế-chung)
5. [Supervised Binning Methods](#5-supervised-binning-methods)
   - 5.1 [Isotonic Regression Binner](#51-isotonic-regression-binner)
   - 5.2 [Quantile Monotonic Binner](#52-quantile-monotonic-binner)
   - 5.3 [Decision Tree Binner](#53-decision-tree-binner)
   - 5.4 [ChiMerge Binner](#54-chimerge-binner)
   - 5.5 [MDLP Binner](#55-mdlp-binner)
   - 5.6 [Spearman Binner](#56-spearman-binner)
   - 5.7 [Log-Odds (Scorecard) Binner](#57-log-odds-scorecard-binner)
   - 5.8 [KS-Optimal Binner](#58-ks-optimal-binner)
6. [Unsupervised Binning Methods](#6-unsupervised-binning-methods)
   - 6.1 [Equal-Width Binner](#61-equal-width-binner)
   - 6.2 [Jenks Natural Breaks Binner](#62-jenks-natural-breaks-binner)
7. [So sánh các phương pháp](#7-so-sánh-các-phương-pháp)
8. [Hướng dẫn sử dụng](#8-hướng-dẫn-sử-dụng)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Tổng quan & Động lực

Binning (discretization) là bước tiền xử lý quan trọng trong xây dựng **scorecard tín dụng** và các mô hình logistic regression tuyến tính. Mục tiêu là chuyển biến liên tục $X$ thành biến phân nhóm, sau đó thay thế mỗi nhóm bằng giá trị **WOE** tương ứng.

**Tại sao cần monotonic binning?**

Trong thực tế tín dụng, có kỳ vọng kinh doanh rõ ràng:

- Thu nhập **càng cao** → khả năng trả nợ **càng tốt** (xác suất bad giảm dần)
- Số ngày quá hạn **càng cao** → rủi ro **càng lớn** (xác suất bad tăng dần)

Một binning không monotonic (event rate lên xuống lộn xộn) sẽ:

1. Vi phạm kỳ vọng kinh doanh → khó giải thích với stakeholder
2. Gây ra WOE không đơn điệu → logistic regression có thể bị nhiễu
3. Là dấu hiệu overfit trên training set

**Định nghĩa chính thức:** Gọi $b_1, b_2, \ldots, b_K$ là $K$ bins sắp xếp theo giá trị $X$ tăng dần. Binning là **monotonic ascending** nếu:

$$\text{EventRate}(b_1) \leq \text{EventRate}(b_2) \leq \cdots \leq \text{EventRate}(b_K)$$

và **monotonic descending** nếu bất đẳng thức đảo chiều.

---

## 2. Nền tảng lý thuyết

### 2.1 Weight of Evidence (WOE)

WOE được giới thiệu bởi Siddiqi (2006) trong bối cảnh xây dựng scorecard tín dụng. Với bin thứ $i$, WOE được định nghĩa là:

$$\text{WOE}_i = \ln\!\left(\frac{P(\text{Event} \mid X \in b_i)}{P(\text{Non-Event} \mid X \in b_i)}\right) = \ln\!\left(\frac{n_{E,i} / N_E}{n_{N,i} / N_N}\right)$$

Trong đó:
- $n_{E,i}$ — số bad (event) trong bin $i$
- $n_{N,i}$ — số good (non-event) trong bin $i$
- $N_E = \sum_i n_{E,i}$ — tổng số bad
- $N_N = \sum_i n_{N,i}$ — tổng số good

**Ý nghĩa:**
- $\text{WOE}_i > 0$: bin $i$ có tỷ lệ bad cao hơn trung bình → **rủi ro cao**
- $\text{WOE}_i < 0$: bin $i$ có tỷ lệ bad thấp hơn trung bình → **rủi ro thấp**
- $\text{WOE}_i = 0$: bin $i$ có tỷ lệ bad đúng bằng trung bình

> **Lưu ý kỹ thuật:** Thêm hằng số $\varepsilon = 10^{-9}$ để tránh $\ln(0)$ khi $n_{E,i} = 0$ hoặc $n_{N,i} = 0$.

### 2.2 Information Value (IV)

IV (Anderson & Banasik, 1999) đo lường **sức mạnh phân biệt** của toàn bộ biến $X$ đối với target $Y$:

$$\text{IV} = \sum_{i=1}^{K} \left(\frac{n_{E,i}}{N_E} - \frac{n_{N,i}}{N_N}\right) \cdot \text{WOE}_i$$

Thang đánh giá IV (Siddiqi, 2006):

| IV | Đánh giá |
|---|---|
| $< 0.02$ | Vô dụng |
| $0.02$ – $0.1$ | Yếu |
| $0.1$ – $0.3$ | Trung bình |
| $0.3$ – $0.5$ | Tốt |
| $> 0.5$ | Nghi ngờ overfit |

IV có quan hệ với **Kullback-Leibler divergence** (thường gọi là KL divergence hay relative entropy):

$$\text{IV} = D_{KL}(P_E \| P_N) + D_{KL}(P_N \| P_E) = \sum_i p_{E,i} \ln\frac{p_{E,i}}{p_{N,i}} + \sum_i p_{N,i} \ln\frac{p_{N,i}}{p_{E,i}}$$

Đây là **symmetric KL divergence** (còn gọi là J-divergence), đo khoảng cách giữa phân phối bad và good.

### 2.3 Ràng buộc Monotonicity

Hướng monotonic được phát hiện tự động qua **Spearman rank correlation** (Spearman, 1904):

$$\rho_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

Trong đó $d_i$ là hiệu của hai rank. Nếu $\rho_s \geq 0$: ascending; nếu $\rho_s < 0$: descending.

---

## 3. Kiến trúc Package

```
binning_process/
├── __init__.py
├── compare.py               # compare_methods(), ALL_METHODS, METHOD_CONFIG
├── report.py                # generate_compare_report() — HTML report nhiều feature
│
├── core/
│   ├── base.py              # BaseBinner — preprocessing, fit, transform, plot
│   ├── utils.py             # detect_direction, compute_woe_iv_table, compute_psi, ...
│   └── merge_process.py     # MergeTrace, enforce_monotonic_traced
│
├── supervised/              # Binners dùng target y
│   ├── isotonic.py
│   ├── quantile.py
│   ├── decision_tree.py
│   ├── chimerge.py
│   ├── mdlp.py
│   ├── spearman.py
│   ├── scorecard.py
│   └── ks_optimal.py
│
└── unsupervised/            # Binners không dùng target y
    ├── equal_width.py
    └── jenks.py
```

**Nguyên tắc thiết kế:**

- `core/` không import từ `supervised/` hay `unsupervised/` → không circular import
- Mỗi Binner chỉ implement `_find_cuts(x, y) → List[float]`; toàn bộ logic còn lại kế thừa từ `BaseBinner`
- `compare.py` là nơi duy nhất biết tất cả methods → thêm method mới chỉ cần thêm entry vào `METHOD_CONFIG` / `ALL_METHODS`

**Luồng `fit()` tổng quát:**

```
BaseBinner.fit(x, y)
    │
    ├─ 1. _preprocess()          Tách missing/special, cap outliers
    │
    ├─ 2. detect_direction()     Spearman → ascending | descending
    │
    ├─ 3. _find_cuts()           [SUBCLASS] đề xuất cut-points ban đầu
    │
    ├─ 4. enforce_monotonic_traced()   Gộp vi phạm, lưu MergeTrace
    │
    └─ 5. compute_woe_iv_table() Tính WOE/IV cho cuts cuối
```

---

## 4. Monotonic Merge — Cơ chế chung

Tất cả Binner đều dùng chung cơ chế **Adjacent Merge** để enforce monotonicity sau khi `_find_cuts()` đề xuất cut-points ban đầu.

**Thuật toán Adjacent Merge:**

```
Input:  cuts = [c₁, c₂, ..., cₘ],  direction ∈ {ascending, descending}
Output: cuts đã monotonic

Repeat:
    Tính event_rate[i] cho từng bin i
    Tìm i* = chỉ số đầu tiên vi phạm:
        ascending:  event_rate[i] > event_rate[i+1]
        descending: event_rate[i] < event_rate[i+1]
    If không tìm thấy i* → DỪNG (đã monotonic)
    Xóa cuts[i*]  (gộp bin i* và i*+1)
Until monotonic hoặc hết cuts
```

**Tính đúng đắn:** Mỗi vòng lặp xóa đúng 1 cut-point, nên sau tối đa $m$ vòng (với $m$ là số cuts ban đầu) sẽ đạt được monotonicity. Thuật toán luôn kết thúc.

**Phức tạp tính toán:** $O(m^2 \cdot n)$ trong trường hợp xấu nhất, với $m$ là số bins ban đầu và $n$ là số quan sát. Thực tế $m \leq 50$ nên rất nhanh.

**Trace & Visualize:** Hàm `enforce_monotonic_traced()` lưu toàn bộ lịch sử vào object `MergeTrace`, cho phép vẽ lại từng bước:

```python
cuts_final, trace = enforce_monotonic_traced(cuts, x, y, direction="descending")
trace.summary()           # Bảng log từng bước
trace.plot_steps()        # Grid tất cả bước (Event Rate | WOE | N samples)
trace.plot_before_after() # So sánh trước / sau
```

---

## 5. Supervised Binning Methods

### 5.1 Isotonic Regression Binner

**File:** `supervised/isotonic.py` &nbsp;|&nbsp; **Class:** `IsotonicBinner`

**Tham khảo:** Barlow et al. (1972), *Statistical Inference under Order Restrictions*. Wiley.

#### Ý tưởng

Fit một hàm bậc thang đơn điệu (isotonic) qua các event rate bin nhỏ, tìm điểm gãy của bậc thang làm cut-points. Isotonic Regression giải bài toán tối ưu sau (trường hợp ascending):

$$\hat{f} = \arg\min_{f \text{ non-decreasing}} \sum_{i=1}^{n} w_i (y_i - f(x_i))^2$$

Lời giải tối ưu được tìm bằng thuật toán **Pool Adjacent Violators (PAV)** với độ phức tạp $O(n)$ (Ayer et al., 1955).

#### Thuật toán

1. Chia $X$ thành $N_{\text{init}}$ bins bằng quantile
2. Tính event rate $\bar{y}_i$ và trọng số $w_i = |b_i|$ cho từng bin
3. Fit Isotonic Regression: $\hat{e}_i = \text{ISO}(\bar{y}_1, \ldots, \bar{y}_{N_{\text{init}}};\ w_1, \ldots, w_{N_{\text{init}}})$
4. Cut-points = trung điểm giữa các cặp bin liền kề có $|\hat{e}_i - \hat{e}_{i+1}| > \varepsilon$

#### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `n_init_bins` | 20 | Số bins quantile ban đầu |
| `max_bins` | 8 | Số bins tối đa sau merge |

---

### 5.2 Quantile Monotonic Binner

**File:** `supervised/quantile.py` &nbsp;|&nbsp; **Class:** `QuantileMonotonicBinner`

**Tham khảo:** Mays (2001), *Handbook of Credit Scoring*. Glenlake Publishing.

#### Ý tưởng

1. Khởi tạo bin bằng quantile_cuts
2. Chuyển sang `enforce_monotonic_traced()`

**Pre-merge** giúp tránh event rate bất ổn định ở bins quá nhỏ, trước khi enforce monotonic.

---

### 5.3 Decision Tree Binner

**File:** `supervised/decision_tree.py` &nbsp;|&nbsp; **Class:** `DecisionTreeBinner`

**Tham khảo:** Breiman et al. (1984), *Classification and Regression Trees*. Wadsworth.

#### Ý tưởng

Dùng cây quyết định nông (shallow) để tìm cut-points tối ưu theo tiêu chí **Gini impurity**. Tại mỗi node, cây tìm ngưỡng $c^*$ tối thiểu hoá:

$$c^* = \arg\min_{c}\ \frac{n_L}{n} G(y_L) + \frac{n_R}{n} G(y_R)$$

Trong đó $G(y) = 2p(1-p)$ là Gini impurity, $p$ là event rate, $n_L, n_R$ là số mẫu hai nhánh.

#### Gini impurity cho bài toán binary

$$G = 1 - p^2 - (1-p)^2 = 2p(1-p)$$

$G = 0$ khi nhóm thuần (toàn bad hoặc toàn good). $G = 0.5$ khi 50% bad / 50% good.

**Weighted Gini** sau khi chia:

$$G_{\text{split}} = \frac{n_L}{n} G(p_L) + \frac{n_R}{n} G(p_R)$$

Cut-point tốt → $G_{\text{split}}$ nhỏ → hai nhóm con tách biệt rõ.

#### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `max_depth` | 4 | Độ sâu tối đa của cây |
| `min_samples_leaf_ratio` | 0.05 | Tỷ lệ mẫu tối thiểu mỗi leaf |

---

### 5.4 ChiMerge Binner

**File:** `supervised/chimerge.py` &nbsp;|&nbsp; **Class:** `ChiMergeBinner`

**Tham khảo:** Kerber (1992), "ChiMerge: Discretization of Numeric Attributes." *AAAI*, pp. 123–128.

#### Ý tưởng

Bắt đầu từ $N_{\text{init}}$ bins nhỏ, lặp lại gộp cặp kề nhau có **Chi-square nhỏ nhất** (tức là giống nhau nhất về phân phối bad/good).

#### Chi-square test cho 2 bins kề nhau

Bảng contingency $2 \times 2$:

|  | Bad | Good |
|---|---|---|
| Bin $i$ | $n_{E,i}$ | $n_{N,i}$ |
| Bin $i+1$ | $n_{E,i+1}$ | $n_{N,i+1}$ |

$$\chi^2 = \sum_{r,c} \frac{(O_{rc} - E_{rc})^2}{E_{rc}}$$

Với $E_{rc} = \dfrac{(\text{row total}_r)(\text{col total}_c)}{\text{grand total}}$.

Phân phối $\chi^2$ với $df = (2-1)(2-1) = 1$ bậc tự do. Ngưỡng dừng: $\chi^2_{\alpha, 1}$ với $\alpha = 0.05$ (mặc định).

#### Điều kiện dừng

Dừng khi **một trong hai** điều kiện thoả mãn:
- Đã đạt `max_bins`
- $\min_i \chi^2_i \geq \chi^2_{0.05, 1} \approx 3.84$ (tất cả bins đủ khác biệt)

#### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `n_init_bins` | 50 | Số bins khởi tạo (nhiều hơn để tìm cuts tốt hơn) |
| `confidence_level` | 0.05 | Mức ý nghĩa thống kê $\alpha$ |

---

### 5.5 MDLP Binner

**File:** `supervised/mdlp.py` &nbsp;|&nbsp; **Class:** `MDLPBinner`

**Tham khảo:** Fayyad & Irani (1993), "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning." *IJCAI*, pp. 1022–1027.

#### Ý tưởng

Recursive binary splitting tối đa hoá **Information Gain**, dừng theo nguyên lý **Minimum Description Length (MDL)**. MDL phát biểu rằng: một cut-point chỉ nên thêm vào nếu "lợi ích thông tin" vượt quá "chi phí mô tả" cut-point đó.

#### Entropy

$$H(S) = -p \log_2 p - (1-p) \log_2(1-p)$$

Với $p$ là event rate trong tập $S$.

#### Information Gain

$$\text{IG}(S, c) = H(S) - \frac{|S_L|}{|S|} H(S_L) - \frac{|S_R|}{|S|} H(S_R)$$

Trong đó $S_L = \{x \in S : x \leq c\}$, $S_R = \{x \in S : x > c\}$.

#### Điều kiện dừng MDL (Fayyad & Irani, 1993)

Dừng chia tại $c$ nếu:

$$\text{IG}(S, c) \leq \frac{\log_2(|S| - 1) + \Delta}{|S|}$$

Trong đó:

$$\Delta = \log_2(3^k - 2) - \left[k \cdot H(S) - k_1 \cdot H(S_L) - k_2 \cdot H(S_R)\right]$$

$k$, $k_1$, $k_2$ là số giá trị phân biệt của $Y$ trong $S$, $S_L$, $S_R$ tương ứng. Với bài toán binary ($k \leq 2$), $\Delta$ thường nhỏ, khiến tiêu chí khá chặt.

---

### 5.6 Spearman Binner

**File:** `supervised/spearman.py` &nbsp;|&nbsp; **Class:** `SpearmanBinner`

**Tham khảo:** Spearman (1904), "The Proof and Measurement of Association between Two Things." *American Journal of Psychology*, 15(1), 72–101.

#### Ý tưởng

Thay vì đề xuất cuts rồi sửa vi phạm, Spearman Binner **trực tiếp tối ưu** mức độ monotonicity bằng greedy search. Mục tiêu là tối đa hoá $|\rho_s(\text{bin\_index}, \text{event\_rate})|$.

**Spearman rank correlation:**

$$\rho_s = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}$$

Với $d_i = \text{rank}(x_i) - \text{rank}(y_i)$. Giá trị $\rho_s \in [-1, 1]$. $|\rho_s| = 1$ ↔ perfectly monotonic.

#### Thuật toán

```
Input: n_init_bins quantile cuts
While len(cuts) >= max_bins:
    For each cut cᵢ:
        Tính ρ_s khi xóa cᵢ  →  score_i
    Xóa cᵢ* có score_i* lớn nhất (hoặc giảm ít nhất)
```

Đây là **greedy forward deletion** — ở mỗi bước chọn cut nào khi xóa gây hại ít nhất cho monotonicity.

---

### 5.7 Log-Odds (Scorecard) Binner

**File:** `supervised/scorecard.py` &nbsp;|&nbsp; **Class:** `LogOddsBinner`

**Tham khảo:** Siddiqi (2006), *Credit Risk Scorecards*. Wiley. &nbsp;|&nbsp; Thomas, Edelman & Crook (2002), *Credit Scoring and its Applications*. SIAM.

#### Ý tưởng

Scorecard tuyến tính ánh xạ WOE sang **điểm số** (score) theo công thức:

$$\text{Score} = A - B \cdot \ln(\text{Odds})$$

Với $A$, $B$ là hằng số calibration. Để điểm số tăng đều và có ý nghĩa, các bins nên **đều nhau trên thang log-odds** (không phải thang $X$ hay event rate).

**Log-odds** của một quan sát:

$$\text{LogOdds}(x) = \ln\!\left(\frac{p(x)}{1 - p(x)}\right)$$

Trong đó $p(x)$ là xác suất bad ước lượng bằng Isotonic Regression.

#### Thuật toán

1. Fit Isotonic Regression để ước lượng $\hat{p}(x)$ đơn điệu
2. Tính $\text{LogOdds}(x) = \ln(\hat{p}(x) / (1 - \hat{p}(x)))$
3. Chia đều thang log-odds thành `max_bins` khoảng bằng nhau:
   $$c_k^{LO} = \text{LogOdds}_{\min} + k \cdot \frac{\text{LogOdds}_{\max} - \text{LogOdds}_{\min}}{\text{max\_bins}}, \quad k = 1, \ldots, K-1$$
4. Tìm $x$ tương ứng với mỗi $c_k^{LO}$ → cut-points trên thang $X$

**Ưu điểm:** WOE của từng bin xấp xỉ đều nhau trên thang log-odds → scorecard có tính tuyến tính đẹp, điểm số tăng đều.

---

### 5.8 KS-Optimal Binner

**File:** `supervised/ks_optimal.py` &nbsp;|&nbsp; **Class:** `KSOptimalBinner`

**Tham khảo:** Kolmogorov (1933); Smirnov (1948). Ứng dụng trong credit scoring: Thomas et al. (2002), *Credit Scoring and its Applications*. SIAM.

#### Ý tưởng

**Kolmogorov-Smirnov (KS) statistic** đo khoảng cách tối đa giữa CDF của bad và CDF của good:

$$\text{KS} = \max_x \left| F_{\text{Bad}}(x) - F_{\text{Good}}(x) \right|$$

Cut-point $c^*$ tối ưu KS là điểm mà tại đó hai CDF cách xa nhau nhất:

$$c^* = \arg\max_c \left| \frac{\sum_{x \leq c} y_i}{N_E} - \frac{\sum_{x \leq c} (1-y_i)}{N_N} \right|$$

#### Thuật toán (Greedy Recursive)

```
Procedure RecursiveKS(x, y, cuts, depth):
    If depth ≥ max_bins - 1: return
    If |x| < min_samples: return
    
    candidates ← percentiles P₁₀, P₁₅, ..., P₉₀ của x
    c* ← argmax_{c ∈ candidates} KS(x, y, c)
    
    cuts.append(c*)
    RecursiveKS(x[x ≤ c*], y[x ≤ c*], cuts, depth+1)
    RecursiveKS(x[x > c*], y[x > c*], cuts, depth+1)
```

**Tại sao KS khác IV?**
- IV tối ưu hoá **tổng** khả năng phân biệt trên toàn bộ khoảng giá trị
- KS tối ưu hoá **điểm phân biệt tốt nhất tại 1 ngưỡng**

KS phù hợp khi mục tiêu là reject/approve tại 1 cutoff duy nhất; IV phù hợp cho scorecard dùng toàn bộ phổ điểm.

---

## 6. Unsupervised Binning Methods

Các phương pháp unsupervised **không dùng target $y$** khi tìm cuts. Sau khi tìm cuts, vẫn dùng `enforce_monotonic_traced()` để đảm bảo WOE đơn điệu. Tuy nhiên nếu signal của $X$ yếu, việc enforce monotonic có thể gộp quá nhiều bins.

### 6.1 Equal-Width Binner

**File:** `unsupervised/equal_width.py` &nbsp;|&nbsp; **Class:** `EqualWidthBinner`

#### Ý tưởng

Chia đều **khoảng giá trị** $[\min X, \max X]$ thành `max_bins` phần bằng nhau:

$$c_k = \min(X) + k \cdot \frac{\max(X) - \min(X)}{\text{max\_bins}}, \quad k = 1, \ldots, K-1$$

**Ưu điểm:** Đơn giản, cut-points có ý nghĩa trực quan (ví dụ: thu nhập 0–20tr, 20–40tr, ...).

**Nhược điểm:** Nếu phân phối lệch (skewed), một số bins sẽ rất ít mẫu. Sau khi outlier capping, vấn đề này giảm bớt.

**Khi nào dùng:** Khi cần bins có biên rõ ràng, dễ hiểu với người dùng cuối; không quan tâm đến phân phối đều mẫu.

---

### 6.2 Jenks Natural Breaks Binner

**File:** `unsupervised/jenks.py` &nbsp;|&nbsp; **Class:** `JenksNaturalBreaksBinner`

**Tham khảo:** Fisher (1958), "On Grouping for Maximum Homogeneity." *Journal of the American Statistical Association*, 53(284), 789–798. &nbsp;|&nbsp; Jenks (1967), *The Data Model Concept in Statistical Mapping*. International Yearbook of Cartography.

#### Ý tưởng

Tìm $K$ bins sao cho:
- **Variance trong bin** (Within-Class Variance, WCV) nhỏ nhất
- **Variance giữa các bins** (Between-Class Variance, BCV) lớn nhất

Tương đương bài toán tối ưu:

$$\min_{\text{partition}} \sum_{k=1}^{K} \sum_{x_i \in b_k} (x_i - \bar{x}_{b_k})^2$$

Đây là bài toán phân cụm 1 chiều (1D k-means), có lời giải tối ưu bằng **Dynamic Programming** (Fisher, 1958).

#### Thuật toán Fisher-Jenks (DP)

Định nghĩa $\text{SSW}(l, r)$ = sum of squared deviations đoạn $[l, r)$:

$$\text{SSW}(l, r) = \sum_{i=l}^{r-1} x_i^2 - \frac{\left(\sum_{i=l}^{r-1} x_i\right)^2}{r - l}$$

Dùng **prefix sum** để tính $O(1)$ mỗi đoạn:

$$\text{SSW}(l, r) = S_2[r] - S_2[l] - \frac{(S_1[r] - S_1[l])^2}{r - l}$$

**Recurrence:**

$$V[i][j] = \min_{l < i} \left\{ V[l][j-1] + \text{SSW}(l, i) \right\}$$

Với $V[i][j]$ = tổng WCV tối thiểu khi chia $i$ điểm đầu thành $j$ bins.

**Độ phức tạp:** $O(n^2 K)$ thời gian, $O(nK)$ không gian. Dùng sampling khi $n > 2000$.

---

## 7. So sánh các phương pháp

### Theo đặc điểm kỹ thuật

| Method | Nhóm | Tối ưu hoá | Cuts từ đâu | Độ phức tạp |
|---|---|---|---|---|
| Isotonic | Supervised | MSE (bậc thang đơn điệu) | PAV algorithm | $O(n)$ |
| Quantile | Supervised | Đều mẫu | Percentile | $O(n \log n)$ |
| DecisionTree | Supervised | Gini impurity | Tree splits | $O(n \log^2 n)$ |
| ChiMerge | Supervised | Chi-square | Bottom-up merge | $O(K^2)$ |
| MDLP | Supervised | Information Gain + MDL | Recursive split | $O(n \log n)$ |
| Spearman | Supervised | $|\rho_s|$ | Greedy deletion | $O(K^2 n)$ |
| LogOdds | Supervised | Đều log-odds | Isotonic + linspace | $O(n)$ |
| KSOptimal | Supervised | KS statistic | Recursive KS | $O(K n)$ |
| EqualWidth | Unsupervised | Đều khoảng giá trị | Linspace | $O(1)$ |
| JenksBreaks | Unsupervised | Within-class variance | Fisher DP | $O(n^2 K)$ |

### Theo mục tiêu sử dụng

| Mục tiêu | Method khuyến nghị |
|---|---|
| Giải thích cho stakeholder phi kỹ thuật | `QuantileMonotonic`, `DecisionTree` |
| Tối đa IV cho scorecard | `ChiMerge`, `MDLP`, `Spearman` |
| Scorecard tuyến tính đẹp | `LogOdds` |
| Tìm ngưỡng reject/approve | `KSOptimal` |
| Bins có biên giá trị rõ ràng | `EqualWidth` |
| Phân cụm tự nhiên trong dữ liệu | `JenksBreaks` |
| Baseline nhanh | `QuantileMonotonic` |

### Ưu / nhược điểm tóm tắt

**Isotonic:** Lý thuyết vững chắc, đảm bảo monotonic từ đầu. Nhạy với nhiễu nếu `n_init_bins` lớn.

**QuantileMonotonic:** Đơn giản nhất, dễ explain. Cuts không phản ánh signal của $Y$.

**DecisionTree:** Cuts tối ưu theo Gini. Có thể bị phân mảnh nếu `max_depth` lớn.

**ChiMerge:** Có cơ sở thống kê rõ ràng. Cần dữ liệu đủ lớn để $\chi^2$ ổn định.

**MDLP:** Tự động kiểm soát số bins, tránh overfit. Tiêu chí dừng đôi khi quá chặt → ít bins.

**Spearman:** Trực tiếp tối ưu monotonicity. Greedy không đảm bảo tối ưu toàn cục.

**LogOdds:** Scorecard đẹp nhất về tính tuyến tính. Phụ thuộc vào chất lượng Isotonic Regression.

**KSOptimal:** Tốt cho bài toán phân loại nhị phân tại 1 ngưỡng. IV có thể thấp hơn các method khác.

**EqualWidth:** Nhanh, trực quan. Không nên dùng khi phân phối lệch mạnh.

**JenksBreaks:** Tìm cấu trúc tự nhiên trong dữ liệu. Chậm với $n$ lớn (cần sampling).

---

## 8. Hướng dẫn sử dụng

### Cài đặt phụ thuộc

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

### Import

```python
from binning_process import generate_compare_report
from binning_process.supervised import ChiMergeBinner, DecisionTreeBinner, IsotonicBinner, MDLPBinner, QuantileMonotonicBinner
from binning_process.compare import compare_methods
```

### Fit 1 biến

```python
binner = ChiMergeBinner(
    feature_name   = "income",
    max_bins       = 6,
    n_init_bins    = 50,
    special_values = [-1],        # -1 = không khai báo
    direction      = "auto",      # tự động phát hiện
)
binner.fit(x_train, y_train)

print(binner.summary())           # WOE/IV table
print(f"IV = {binner.final_iv_:.4f}")
print(f"Monotonic: {binner.is_monotonic()}")

woe_train = binner.transform(x_train)
woe_test  = binner.transform(x_test)

binner.plot()                     # WOE bar + Event Rate line
```

### So sánh tất cả methods

```python
result, fitted = compare_methods(
    x              = x_train,
    y              = y_train,
    feature_name   = "income",
    max_bins       = 6,
    n_init_bins    = 20,
    special_values = [-1],
    # methods = ["ChiMerge", "MDLP", "Spearman"],  # chỉ so sánh 1 số method
)

# Xem chi tiết method tốt nhất
best_name  = result.iloc[0]["Method"]
best_model = fitted[best_name]
best_model.plot()
```

### Xem trace merge

```python
binner.fit(x_train, y_train)

# Xem toàn bộ quá trình gộp bins
binner.trace_.summary()
binner.trace_.plot_steps()        # Grid: Event Rate | WOE | N samples
binner.trace_.plot_before_after() # Trước / sau
binner.trace_.plot_step(2)        # Chi tiết bước 2

# Lấy cuts cuối
print(binner.final_cuts_)
print(binner.trace_.final_cuts)   # Giống nhau
```

### HTML Report nhiều feature

Để so sánh nhiều thuật toán trên nhiều cột (vd 20 feature) và xuất một file HTML có tab theo feature, dùng `generate_compare_report`. Mỗi tab gồm **Overview** (bảng so sánh, Final WOE/Event Rate theo thuật toán, Train vs Valid + PSI nếu có) và **Chi tiết** (plot Init/Algo/Final, plot Merge Steps).

```python
from binning_process import generate_compare_report

# Chỉ train
generate_compare_report(
    df_train=df,
    target_col="y",
    feature_cols=["f1", "f2", ...],  # ví dụ 20 cột
    output_path="compare_report.html",
    verbose=False,
)

# Có train + valid (fit trên train, so sánh final WOE & PSI)
generate_compare_report(
    df_train=df_train,
    df_valid=df_valid,
    target_col="y",
    feature_cols=feature_cols,
    output_path="compare_report.html",
    verbose=False,
)
```

### Thêm method mới

```python
# 1. Tạo file supervised/my_binner.py
from ..core.base import BaseBinner

class MyBinner(BaseBinner):
    def _find_cuts(self, x, y):
        # Logic của bạn → trả về List[float]
        ...

# 2. Thêm vào compare.py
ALL_METHODS["MyBinner"] = MyBinner
```

---

## 9. Tài liệu tham khảo

1. **Anderson, R. & Banasik, J.** (1999). *Credit Scoring and the Control of Bad Debt*. Chapman & Hall.

2. **Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T., & Silverman, E.** (1955). "An empirical distribution function for sampling with incomplete information." *Annals of Mathematical Statistics*, 26(4), 641–647.

3. **Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D.** (1972). *Statistical Inference under Order Restrictions*. Wiley.

4. **Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J.** (1984). *Classification and Regression Trees*. Wadsworth.

5. **Fayyad, U. M. & Irani, K. B.** (1993). "Multi-interval discretization of continuous-valued attributes for classification learning." *Proceedings of IJCAI-93*, pp. 1022–1027.

6. **Fisher, W. D.** (1958). "On grouping for maximum homogeneity." *Journal of the American Statistical Association*, 53(284), 789–798.

7. **Jenks, G. F.** (1967). "The data model concept in statistical mapping." *International Yearbook of Cartography*, 7, 186–190.

8. **Kerber, R.** (1992). "ChiMerge: Discretization of numeric attributes." *Proceedings of AAAI-92*, pp. 123–128.

9. **Kolmogorov, A. N.** (1933). "Sulla determinazione empirica di una legge di distribuzione." *Giornale dell'Istituto Italiano degli Attuari*, 4, 83–91.

10. **Mays, E.** (2001). *Handbook of Credit Scoring*. Glenlake Publishing.

11. **Siddiqi, N.** (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.

12. **Smirnov, N. V.** (1948). "Table for estimating the goodness of fit of empirical distributions." *Annals of Mathematical Statistics*, 19(2), 279–281.

13. **Spearman, C.** (1904). "The proof and measurement of association between two things." *American Journal of Psychology*, 15(1), 72–101.

14. **Thomas, L. C., Edelman, D. B., & Crook, J. N.** (2002). *Credit Scoring and Its Applications*. SIAM.
