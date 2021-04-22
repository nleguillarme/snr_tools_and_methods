FROM        ubuntu:latest

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timez

# update and install dependencies
RUN         apt-get update \
                && apt-get install -y \
                    software-properties-common \
                    python3-pip \
                    git \
                    vim \
                    wget \
                    gawk

RUN useradd mer

WORKDIR /home/mer

RUN pip3 install --upgrade pip
RUN pip3 install merpy pandas

COPY ncbi.txt /home/mer/ncbi.txt

COPY create_lexicon.py /home/mer/create_lexicon.py

RUN python3 create_lexicon.py

COPY run_merpy.py /home/mer/run_merpy.py
