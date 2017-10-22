FROM andreptb/oracle-java:8

WORKDIR /spv2-dataprep

COPY server/target/scala-2.11/spv2-dataprep-server-assembly-1.2-SNAPSHOT.jar /spv2-dataprep/dataprep.jar

EXPOSE 8080

CMD ["java", "-Xmx8G", "-jar", "dataprep.jar"]
