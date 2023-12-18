import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Other setup Constants
SCALA_VERSION = '2.12'
SPARK_VERSION = '3.5.0'
KAFKA_VERSION = '3.6.0'

packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
    f'org.apache.kafka:kafka-clients:{KAFKA_VERSION}'
]


class SparkProcessor:
    def start(self):
        # Initialize Spark session
        findspark.init()
        spark = SparkSession.builder \
            .master('local') \
            .appName("person-reid") \
            .config("spark.jars.packages", ",".join(packages)) \
            .getOrCreate()

        # Subscribe to multiple Kafka topics
        df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "topic_camera_00, topic_camera_01") \
            .option("startingOffsets", "latest") \
            .load()

        processed_data = df.select(
            col('topic'),
            col('offset'),
            col('value').cast('binary').alias("image")
        )

        # Start the streaming query
        query = processed_data \
            .writeStream \
            .queryName("topic_camera") \
            .trigger(processingTime="0.5 seconds") \
            .outputMode("append") \
            .option("truncate", "true") \
            .format("console") \
            .start()

        # Await termination of the streaming query
        query.awaitTermination()
        spark.stop()


if __name__ == "__main__":
    spark_processor = SparkProcessor()
    spark_processor.start()
