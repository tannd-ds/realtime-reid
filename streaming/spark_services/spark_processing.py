import time
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from realtime_reid.pipeline import Pipeline

# Other setup Constants
SCALA_VERSION = '2.12'
SPARK_VERSION = '3.5.0'
KAFKA_VERSION = '3.6.0'

packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
    f'org.apache.kafka:kafka-clients:{KAFKA_VERSION}'
]


class SparkProcessor:
    def __init__(self) -> None:
        self.pipeline = Pipeline()

        # Initialize Spark session
        findspark.init()
        self.spark = SparkSession.builder \
            .master('local') \
            .appName("person-reid") \
            .config("spark.jars.packages", ",".join(packages)) \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()

        # Subscribe to multiple Kafka topics
        self.df_raw = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "topic_camera_00, topic_camera_01") \
            .option("startingOffsets", "latest") \
            .load()

        self.df = self.df_raw \
            .select(
                col('topic').cast('string'),
                col('offset').cast('long'),
                col('value').cast('binary').alias("image"),
            )

    def start(self) -> None:
        """
        Starts the streaming query to process data and display the
        results in the console.

        Returns:
            None
        """
        # Start the streaming query
        stream_writer = self.df \
            .writeStream \
            .queryName("topic_camera") \
            .trigger(processingTime="0.5 seconds") \
            .outputMode("append") \
            .option("truncate", "False") \
            .format("memory") \

        self.query = stream_writer.start()

    def print_query(self):
        last_offset = -1
        while True:
            df = self.spark.sql(
                f"SELECT * FROM {self.query.name} WHERE offset > {last_offset}"
            ).toPandas()
            df['processed_image'] = df['image'].apply(
                lambda x: self.pipeline.process(x, return_bytes=True)
            )
            if df.shape[0] > 0:
                print(df)
                last_offset = df['offset'].max()
            time.sleep(1)


if __name__ == "__main__":
    spark_processor = SparkProcessor()
    spark_processor.start()
    try:
        spark_processor.print_query()
    except KeyboardInterrupt:
        print("Interupted by user.")

    # Await termination of the streaming query
    spark_processor.query.awaitTermination()
    spark_processor.spark.stop()
