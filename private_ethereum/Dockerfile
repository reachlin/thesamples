FROM ubuntu:18.04
RUN apt-get update && apt install -y software-properties-common
RUN add-apt-repository -y ppa:ethereum/ethereum
RUN apt-get update && apt-get install -y ethereum
RUN mkdir -p /data
WORKDIR /data
COPY genesis.json /root/
