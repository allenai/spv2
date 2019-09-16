FROM ubuntu:18.04

RUN apt-get update -y && apt-get install openjdk-8-jdk -y

WORKDIR /spv2-dataprep

ARG version

COPY server/target/scala-2.11/spv2-dataprep-server-assembly-${version}.jar /spv2-dataprep/dataprep.jar

EXPOSE 8080

CMD ["java", "-Xmx3G", "-jar", "dataprep.jar"]
