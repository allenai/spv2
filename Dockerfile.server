FROM ubuntu:16.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /spv2

# Install python dependencies
RUN apt-get update -y && \
    apt-get install apt-utils -y && \
    apt-get upgrade -y && \
    apt-get install git python3 python3-pip python3-cffi unzip wget -y && \
    pip3 install --upgrade pip

# Install pip dependencies
COPY requirements.in .
RUN pip3 install -r requirements.in

# Copy model
COPY model/ model/

# Copy and build the stringmatch module
COPY stringmatch/ stringmatch/
RUN cd stringmatch && python3 stringmatch_builder.py && cd ..

# Copy the code itself
COPY *.py ./
COPY *.sh ./

# Install an optimized version of tensorflow
COPY tensorflow-cpu/ tensorflow-cpu/
RUN pip3 uninstall -y tensorflow && pip3 install tensorflow-cpu/tensorflow-1.3.1-cp35-cp35m-linux_x86_64.whl

# Install supervisor
RUN apt-get install software-properties-common python-software-properties supervisor -y && \
    mkdir -p /var/log/supervisor

# Install java
RUN apt-get install openjdk-8-jdk -y
# Better perf from oracle jdk but this dependency just 404'd on 2018-10-16
# Same was true when I tried ppa:linuxuprising/java for java10
#     add-apt-repository ppa:webupd8team/java -y && \
#     apt-get update -y && \
#     echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
#     echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \
#     apt-get install oracle-java8-installer oracle-java8-set-default -y

COPY supervisord.conf supervisord.conf

# Copy the dataprep jar file
COPY dataprep/server/target/scala-2.11/spv2-dataprep-server-assembly-*.jar /spv2-dataprep/dataprep.jar

EXPOSE 8081

CMD ["/usr/bin/supervisord", "-n", "-e", "debug", "-c", "supervisord.conf"]
