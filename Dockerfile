# Apache Airflow on Python 3.11 (modern sklearn / xgboost / pyspark compatible).
FROM apache/airflow:2.7.3-python3.11

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh && \
    ln -sfn "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /opt/java && \
    echo "Resolved JAVA_HOME -> $(readlink -f /opt/java)" && \
    /opt/java/bin/java -version

ENV JAVA_HOME=/opt/java
ENV PATH=$PATH:/opt/java/bin
# Make the project importable (utils package)
ENV PYTHONPATH=/opt/airflow

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
