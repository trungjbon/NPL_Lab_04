# 1. Các bước triển khai
## Bước 1: Khởi tạo Spark Session
```
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
```
## Bước 2: Nạp và tiền xử lý dữ liệu
- Chuyển nhãn cảm xúc từ -1/1 sang 0/1 để mô hình học dễ hơn:
```
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
```
- Làm sạch văn bản bằng cách loại bỏ URL, ký tự HTML, ký tự đặc biệt:
```
df = df.withColumn("text", trim(lower(
        regexp_replace(col("text"), r"http\S+|www\S+|<.*?>|[^a-zA-Z0-9\s]", " ")
        ))
    )
```
- Xóa các dòng trống hoặc giá trị null:
```
df = df.dropna(subset=["text", "sentiment"])
df = df.filter(length(col("text")) > 0)
```
