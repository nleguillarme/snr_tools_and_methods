FROM        ubuntu:xenial

# update and install dependencies
RUN         apt-get update \
                && apt-get install -y \
                    software-properties-common \
                    wget

RUN useradd gnfinder

WORKDIR /home/gnfinder

RUN wget https://github.com/gnames/gnfinder/releases/download/v0.11.1/gnfinder-v0.11.1-linux.tar.gz
RUN tar -xvf gnfinder-v0.11.1-linux.tar.gz
RUN chmod u+x gnfinder

ENV PATH="/home/gnfinder:${PATH}"
