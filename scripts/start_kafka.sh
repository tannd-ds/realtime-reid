#!bin/bash
echo -e "\033[37;5mStarting zookeeper...\033[0m";
while ! nc -z localhost 2181; 
    do 
        sleep 0.5; 
        ~/kafka/bin/zookeeper-server-start.sh -daemon ~/kafka/config/zookeeper.properties
    done;

if nc -z localhost 2181; then
    echo -e "\033[32;1;7mZookeeper started successfully.\033[0m";
else
    echo "Zookeeper failed to start";
    exit 1;
fi

echo -e "Starting kafka... This may take a while, grab a coffee and relax...\033[0m";
while ! nc -z localhost 9092; 
    do 
        sleep 2.0; 
        ~/kafka/bin/kafka-server-start.sh -daemon ~/kafka/config/server.properties
    done;

if nc -z localhost 9092; then
    echo -e "\033[32;1;7mKafka started successfully.\033[0m";
else
    echo "Kafka failed to start";
    exit 1;
fi
