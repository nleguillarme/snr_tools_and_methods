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

RUN useradd taxonerd

WORKDIR /home/taxonerd
RUN mkdir input
RUN mkdir output

RUN pip3 install --upgrade pip
RUN pip3 install https://github.com/nleguillarme/taxonerd/archive/refs/tags/v1.1.0.tar.gz
RUN pip3 install https://github.com/nleguillarme/taxonerd/releases/download/v1.1.0/en_ner_eco_md-1.0.0.tar.gz
RUN pip3 install https://github.com/nleguillarme/taxonerd/releases/download/v1.1.0/en_ner_eco_biobert-1.0.0.tar.gz

