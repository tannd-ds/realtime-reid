import threading
import numpy as np
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType
from realtime_reid.pipeline import Pipeline


def start_spark():
    SCALA_VERSION = '2.12'
    SPARK_VERSION = '3.5.0'
    KAFKA_VERSION = '3.6.0'

    packages = [
        f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
        f'org.apache.kafka:kafka-clients:{KAFKA_VERSION}'
    ]

    findspark.init()

    def spark_streaming_thread():
        reid_pipeline = Pipeline()

        # Initialize Spark session
        spark = SparkSession.builder \
            .master('local') \
            .appName("person-reid") \
            .config("spark.jars.packages", ",".join(packages)) \
            .getOrCreate()

        spark.conf.set("spark.sql.shuffle.partitions", "20")

        # Define the Kafka parameters
        kafka_params = {
            "kafka.bootstrap.servers": "localhost:9092",
            "subscribe": "topic_camera_00, topic_camera_01"
        }

        # Create a streaming DataFrame with Kafka source
        df = spark.readStream \
            .format("kafka") \
            .options(**kafka_params) \
            .option("startingOffsets", "latest") \
            .load()

        df = df.withColumn("value", df["value"].cast(BinaryType()))

        @udf(BinaryType())
        def process_frame(value):
            # Buffer -> np.array -> Bytes
            frame = np.frombuffer(value, dtype=np.uint8)
            frame = frame.tobytes()

            frame_bytes = reid_pipeline.process(frame, return_bytes=True)
            return frame_bytes

        # Apply the UDF to process the frames
        processed_df = df \
            .selectExpr("CAST(key AS STRING)",
                        "CAST(topic as STRING)",
                        "value") \
            .withColumn("value", process_frame("value"))

        # Define the Kafka parameters for writing
        write_params = [
            {
                "kafka.bootstrap.servers": "localhost:9092",
                "topic": "processed_topic_1"
            },
            {
                "kafka.bootstrap.servers": "localhost:9092",
                "topic": "processed_topic_2"
            }
        ]

        # Write the processed frames back to Kafka
        query_topic1 = processed_df \
            .filter("topic = 'topic_camera_00'") \
            .writeStream \
            .format("kafka") \
            .options(**write_params[0]) \
            .option("checkpointLocation", "tmp/" + write_params[0]["topic"]) \
            .outputMode("append") \
            .start()

        query_topic2 = processed_df \
            .filter("topic = 'topic_camera_01'") \
            .writeStream \
            .format("kafka") \
            .options(**write_params[1]) \
            .option("checkpointLocation", "tmp/" + write_params[1]["topic"]) \
            .outputMode("append") \
            .start()

        # Start the streaming context
        query_topic1.awaitTermination()
        query_topic2.awaitTermination()

    thread = threading.Thread(target=spark_streaming_thread)
    thread.start()
    return thread
