FROM        ubuntu:xenial

# update and install dependencies
RUN         apt-get update \
                && apt-get install -y \
                    software-properties-common \
                    python-minimal \
                    git \
                    vim \
                    wget \
                    python-pip
RUN useradd netineti

WORKDIR /home/netineti

RUN git clone https://github.com/dshorthouse/NetiNeti
RUN pip install --upgrade "pip < 21.0"

WORKDIR /home/netineti/NetiNeti

# RUN pip install virtualenv \
#    && virtualenv -p /usr/bin/python2.7 venv \
#    && . venv/bin/activate

RUN wget https://pypi.python.org/packages/50/9e/39418026bf8013bbc2852c7aec3fb21e4339f6cd694934496d67a19b53b8/nltk-2.0.5.tar.gz#md5=c3f77cc1bf6c6bc85db6629b7d23c3b9 \
   && tar -xzf nltk-2.0.5.tar.gz \
   && sed -i 's/http/https/g' nltk-2.0.5/distribute_setup.py \
   && pip install --upgrade setuptools \
   && pip install ./nltk-2.0.5

RUN pip install pyyaml nose

RUN cp config/neti_http_config.cfg.example config/neti_http_config.cfg

RUN python -c "exec(\"import nltk\nnltk.download('punkt')\")"

COPY neti.py /home/netineti/NetiNeti/neti.py
