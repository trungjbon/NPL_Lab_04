# 1. Các bước triển khai
## Bước 1: Khởi tạo Spark Session
```
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
```
## Bước 2: Nạp và tiền xử lý dữ liệu
- Đọc dữ liệu:
```
df = spark.read.csv(data_path, header=True, inferSchema=True)
```
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
## Bước 3.1: Xây dựng mô hình cơ sở (TF-IDF + Logistic Regression)
- Mô hình cơ bản sử dụng:
  - TF-IDF để biểu diễn văn bản dưới dạng vector tần suất có trọng số.
  - Logistic Regression để dự đoán cảm xúc.
- Pipeline bao gồm:
  - Tokenizer: tách từ.
  - StopWordsRemover: loại bỏ từ dừng.
  - HashingTF -> IDF: trích xuất đặc trưng TF-IDF.
  - LogisticRegression: mô hình phân loại tuyến tính.

## Bước 3.2: Xây dựng mô hình cải tiến (CountVectorizer + Naive Bayes)
- Để cải thiện hiệu suất, mô hình mới được sử dụng là:
  - CountVectorizer: chuyển văn bản thành vector tần suất từ (Bag-of-Words).
  - NaiveBayes (multinomial): mô hình xác suất phù hợp với dữ liệu rời rạc.
- Pipeline cải tiến gồm:
  - Tokenizer: tách từ.
  - StopWordsRemover: loại bỏ từ dừng.
  - CountVectorizer: trích xuất đặc trưng tần suất.
  - NaiveBayes: mô hình phân loại xác suất.
 
## Bước 4: Đánh giá mô hình
- Hai chỉ số được sử dụng:
  - Accuracy (Độ chính xác)
  - F1-score (hiệu suất trung bình giữa Precision và Recall)
- Dùng MulticlassClassificationEvaluator để tính các chỉ số này.

# 2. Hướng dẫn chạy mã nguồn
- Yêu cầu cài đặt PySpark
- Các file chạy nằm trong Lab_04/test
  - File "lab4_test.py":
  ```
  python lab4_test.py
  ```
  - Kết quả in ra:
  ```
        Evaluation Metrics:
        Accuracy: 0.5000
        Precision: 0.0000
        Recall: 0.0000
        F1: 0.0000
  ```
  - File "lab4_spark_sentiment_analysis.py":
  ```
  python lab4_spark_sentiment_analysis.py
  ```
  - Kết quả in ra:
  ```
        Total row data:  5791
        +--------------------+---------+-----+
        |                text|sentiment|label|
        +--------------------+---------+-----+
        |Kickers on my wat...|        1|  1.0|
        |user: AAP MOVIE. ...|        1|  1.0|
        |user I'd be afrai...|        1|  1.0|
        |   MNTA Over 12.00  |        1|  1.0|
        |    OI  Over 21.37  |        1|  1.0|
        +--------------------+---------+-----+
        only showing top 5 rows
        25/10/28 17:27:07 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
        Accuracy: 0.7295
        F1-score: 0.7266
        +--------------------+-----+----------+--------------------+
        |                text|label|prediction|         probability|
        +--------------------+-----+----------+--------------------+
        |  ISG An update t...|  0.0|       1.0|[0.26145997867802...|
        |  The rodeo clown...|  0.0|       0.0|[0.99999979918751...|
        | , ES,SPY, Ground...|  0.0|       1.0|[0.05410932380488...|
        | ES, S  PAT TWO, ...|  0.0|       0.0|[0.99866261133793...|
        | PCN doulble top ...|  0.0|       1.0|[0.44652779058409...|
        | also not very he...|  0.0|       1.0|[0.41120871544432...|
        | thinking out lou...|  1.0|       0.0|[0.99993303591410...|
        |"RT @WSJheard: Ca...|  1.0|       1.0|[0.02223734716793...|
        |#ContrAlert Don't...|  0.0|       1.0|[0.00118230879877...|
        |#CoronavirusPande...|  0.0|       0.0|[0.99517446040805...|
        +--------------------+-----+----------+--------------------+
        only showing top 10 rows
  ```
  - File "lab4_improvement_test.py":
  ```
  python lab4_improvement_test.py
  ```
  - Kết quả in ra:
  ```
        Total row data:  5791
        +--------------------+---------+-----+
        |                text|sentiment|label|
        +--------------------+---------+-----+
        |kickers on my wat...|        1|  1.0|
        |user  aap movie  ...|        1|  1.0|
        |user i d be afrai...|        1|  1.0|
        |     mnta over 12 00|        1|  1.0|
        |      oi  over 21 37|        1|  1.0|
        +--------------------+---------+-----+
        only showing top 5 rows
        25/10/28 17:30:25 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
        Accuracy: 0.7953
        F1-score: 0.7920
        +--------------------+-----+----------+--------------------+
        |                text|label|prediction|         probability|
        +--------------------+-----+----------+--------------------+
        |11 59 is next fib...|  0.0|       0.0|[0.61447425908260...|
        |15 16 stocks on w...|  1.0|       1.0|[4.14425193822794...|
        |2020 was supposed...|  0.0|       0.0|[0.99386431848004...|
        |3 tech stocks wit...|  1.0|       1.0|[6.36097584475285...|
        |5 small caps bein...|  0.0|       0.0|[0.72201848778962...|
        |744 96 is a great...|  1.0|       1.0|[0.47945560863605...|
        |a couple biotech ...|  1.0|       1.0|[0.10650038289006...|
        |a few names with ...|  0.0|       0.0|[0.84249147283216...|
        |a q4 2012 operati...|  0.0|       0.0|[0.99331453274966...|
        |a q4 net income 5...|  0.0|       0.0|[0.60362802825100...|
        +--------------------+-----+----------+--------------------+
        only showing top 10 rows
  ```
  
# 3. Phân tích kết quả
- Mô hình cơ sở (Logistic Regression + TF-IDF) cho ra:
  - Accuracy: 0.7295
  - F1-score: 0.7266

- Mô hình cải tiến (Naive Bayes + CountVectorizer) cho ra:
  - Accuracy: 0.7953
  - F1-score: 0.7920
 
- Nhận xét:
  - Mô hình cơ bản (LogisticRegression + TF-IDF) đạt khoảng 72% độ chính xác => Bị hạn chế bởi đặc trưng thưa (sparse features) và dữ liệu nhỏ.
  - Mô hình cải tiến (NaiveBayes + CountVectorizer) đạt 79% độ chính xác, tăng ~7%.

- Nguyên nhân cải thiện:
  - Logistic Regression cần nhiều dữ liệu và dễ overfit hơn với TF-IDF.
  - Naive Bayes phù hợp với dữ liệu văn bản nhỏ và rời rạc (từ, câu ngắn).
  - CountVectorizer tạo đặc trưng phù hợp với giả định độc lập trong NB.
  - Sự kết hợp CountVectorizer + NB đơn giản nhưng hiệu quả hơn trên tập 5k mẫu.
 
# 4. Khó khăn và hướng giải quyết
- Văn bản chứa nhiều nhiễu như URL, ký tự HTML, ký tự đặc biệt -> Dùng regex làm sạch văn bản
