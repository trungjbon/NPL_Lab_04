from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim, length
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main():
    # Initialize Spark Session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    
    # Load Data
    data_path = "Lab_04\\data\\sentiments.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    df = df.dropna(subset=["sentiment"])
    df = df.withColumn("text", trim(lower(
        regexp_replace(col("text"), r"http\S+|www\S+|<.*?>|[^a-zA-Z0-9\s]", " ")
        ))
    )
    df = df.filter(length(col("text")) > 0)

    print("Total row data: ", df.count())
    df.show(5, truncate=80)

    # Build Preprocessing Pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    cv = CountVectorizer(inputCol="filtered_words", outputCol="features")

    # Split Data
    trainingData, testData = df.randomSplit([0.8, 0.2], seed=42)

    # Train the Model
    nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, cv, nb])
    model = pipeline.fit(trainingData)

    # Evaluate the Model
    predictions = model.transform(testData)
    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    predictions.select("text", "label", "prediction", "probability").show(10, truncate=80)

    spark.stop()


if __name__ == "__main__":
    main()