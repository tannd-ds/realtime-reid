import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType
import numpy as np
from realtime_reid.pipeline import Pipeline

reid_pipeline = Pipeline()

# Other setup Constants
SCALA_VERSION = '2.12'
SPARK_VERSION = '3.5.0'
KAFKA_VERSION = '3.6.0'

packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
    f'org.apache.kafka:kafka-clients:{KAFKA_VERSION}'
]


# Initialize Spark session
findspark.init()
spark = SparkSession.builder \
    .master('local') \
    .appName("person-reid") \
    .config("spark.jars.packages", ",".join(packages)) \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Define the Kafka parameters
kafka_params = {
    "kafka.bootstrap.servers": "localhost:9092",
    "subscribe": "topic_camera_01"
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
    .selectExpr("CAST(key AS STRING)", "value") \
    .withColumn("value", process_frame("value"))

# Define the Kafka parameters for writing
kafka_write_params = {
    "kafka.bootstrap.servers": "localhost:9092",
    "topic": "processed_topic"
}

# Write the processed frames back to Kafka
query = processed_df \
    .writeStream \
    .format("kafka") \
    .options(**kafka_write_params) \
    .option("checkpointLocation", "tmp/checkpoint") \
    .outputMode("append") \
    .start()

# Start the streaming context
query.awaitTermination()
