from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main():
    # Initialize Spark Session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    
    # Load Data
    data_path = "Lab_04\\data\\sentiments.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df.dropna(subset=["sentiment"])
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    print("Total row data: ", df.count())
    df.show(5)

    # Build Preprocessing Pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")

    # Split Data
    trainingData, testData = df.randomSplit([0.8, 0.2], seed=42)

    # Train the Model
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
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
