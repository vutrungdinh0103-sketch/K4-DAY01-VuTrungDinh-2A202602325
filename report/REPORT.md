# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):

    > ```json
  > {
  >   "taxonomy_name": "ImageNet-1K",
  >   "rank": 1,
  >   "class_id": 468,
  >   "class_name": "cab",
  >   "score": 0.510915
  > }
  > 
- Record này mô tả toàn ảnh như thế nào?
    > Mô hình đánh giá lớp phù hợp nhất là cab (taxi) với độ tin cậy khoảng 51%.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    > Bộ dữ liệu (taxonomy) dùng để huấn luyện checkpoint dự đoán danh sách lớp.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    > Vì với cùng một ID nhưng ở taxonomy khác nhau, ID đó có thể đại diện cho lớp hoàn toàn khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    > Guideline phải quy định tiêu chí chọn nhãn duy nhất khi ảnh chứa nhiều vật thể.
- Vì sao model score không phải ground truth?
    > Ground truth là “đáp án chuẩn” do con người hoặc bộ dữ liệu cung cấp; model score là “mức độ tự tin” của mô hình đối với dự đoán của chính nó.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):

    > ```json
  > {
  >   "class_name": "bus",
  >   "score": 0.912557,
  >   "bbox_xyxy": [93.17, 187.95, 223.01, 320.91],
  >   "bbox_width": 129.84,
  >   "bbox_height": 132.96
  > }
  > 
- Diễn giải vị trí box bằng lời:
    > Bounding box bắt đầu tại điểm (93, 188) và kết thúc tại (223, 321), với kích thước khoảng 130 × 133 pixel.
- So sánh số prediction ở hai threshold:
    > Threshold thấp cho nhiều dự đoán hơn, threshold cao cho ít dự đoán hơn.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    > Nếu giảm threshold làm tăng độ bao phủ (coverage) nhưng cũng làm tăng khối lượng công việc của reviewer, vì họ phải xác nhận và loại bỏ nhiều dự đoán hơn.
- Đề xuất một quy tắc box chặt:
    > Bounding box phải ôm sát phần nhìn thấy của vật thể, không lấy nền dư và không suy đoán phần bị che khuất.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  > **Với vật thể bị che khuất hoặc cắt mép, guideline cần quy định:**
  >
  > - Che khuất (occlusion): Chỉ vẽ box bao phần nhìn thấy, không suy đoán phần bị khuất.
  > - Cắt mép ảnh (truncation): Box được phép chạm mép ảnh, không vượt ra ngoài khung hình.
  > - Escalation: Nếu lớp hoặc ranh giới vật thể quá mơ hồ, chuyển reviewer/QA quyết định để đảm bảo nhất quán.
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

    > ```json
  > {
  >   "instance_id": "traffic-001",
  >   "class_name": "bus",
  >   "score": 0.925745,
  >   "polygon_xy": [
  >     [148.0, 189.0],
  >     [147.0, 190.0]
  >   ]
  > }
  > ```
- Polygon bổ sung chi tiết gì so với box?
    > Bounding box chỉ cho biết hình chữ nhật bao quanh vật thể, còn polygon mô tả đường viền chính xác của vật thể theo nhiều điểm tọa độ.
- `instance_id` dùng để làm gì và không phải loại ID nào?
    > `instance_id` là ID duy nhất của từng vật thể trong một ảnh (ví dụ traffic-001, traffic-002) để phân biệt nhiều xe cùng lớp. Nó không phải `class_id` và cũng không phải `coco_image_id`.
- Đề xuất một quy tắc biên mask:
    > Mask phải bám sát biên nhìn thấy của vật thể, không lấn nền, không để hở khoảng trống và không vẽ phần bị khuất.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    > + Vùng mờ: bám theo ranh giới nhìn thấy rõ nhất.
    > + Hai vật thể tiếp xúc: mỗi vật thể có một mask riêng, không gộp.
    > + Che khuất: chỉ tô phần nhìn thấy; nếu ranh giới quá mơ hồ thì escalation cho QA/reviewer quyết định.
## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| **Phân loại ảnh** | 1 nhãn duy nhất (`class_id`, `class_name`) | Ảnh có nhiều chủ thể, khó xác định đối tượng chính | Chọn 1 lớp theo đúng guideline | Kiểm tra nhãn có đúng nội dung chính của ảnh |
| **Phát hiện vật thể** | 1 bounding box (`xyxy`) + 1 lớp cho mỗi vật thể | Vật thể nhỏ, chồng lấp, bị che khuất hoặc bị bỏ sót | Vẽ box ôm sát từng vật thể và gán đúng lớp | Kiểm tra đủ số lượng vật thể, đúng lớp và box có chặt không |
| **Instance Segmentation** | 1 mask/polygon + 1 lớp cho mỗi instance | Biên mờ, vật thể tiếp xúc nhau hoặc bị che khuất | Vẽ mask bám sát phần nhìn thấy của vật thể | Kiểm tra mask chính xác, không lấn nền và tách đúng từng instance |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
    > Chỉ sử dụng ảnh và dữ liệu đúng phạm vi được cấp, không sao chép, chia sẻ hoặc lưu trữ ra ngoài hệ thống được phép.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
    > Dừng xử lý ngay và báo cho Reviewer, QA hoặc quản lý dự án (Project Manager) để được hướng dẫn.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
