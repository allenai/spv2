FROM ubuntu:16.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

ARG GITHUB_ACCESS_TOKEN

WORKDIR /spv2

# Install dependencies
RUN apt-get update -y && \
    apt-get install apt-utils -y && \
    apt-get upgrade -y && \
    apt-get install git python3 python3-pip python3-cffi unzip wget -y && \
    pip3 install pip_install_privates

# Install pip dependencies
COPY requirements.in .
RUN pip_install_privates --token=${GITHUB_ACCESS_TOKEN} requirements.in

# Copy model
COPY model/ model/

# Copy and build the stringmatch module
COPY stringmatch/ stringmatch/
RUN cd stringmatch && python3 stringmatch_builder.py && cd ..

# Copy the code itself
COPY db_worker/ db_worker/
COPY *.py ./

# Install an optimized version of tensorflow
COPY tensorflow-cpu/ tensorflow-cpu/
RUN pip3 uninstall -y tensorflow && pip3 install tensorflow-cpu/tensorflow-1.3.1-cp35-cp35m-linux_x86_64.whl

CMD python3 db_worker/db_worker.py
