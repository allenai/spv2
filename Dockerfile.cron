FROM ubuntu:16.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /spv2

# Make setup sane
RUN apt-get update -y && \
    apt-get install apt-utils -y && \
    apt-get upgrade -y && \
    apt-get install curl jq -y

# Install kubectl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl && \
    chmod +x ./kubectl
ENV PATH="./:${PATH}"

# Copy yaml files
COPY *.yaml ./
COPY dataprep/*.yaml ./dataprep/

COPY cron.sh ./

CMD /bin/bash cron.sh
