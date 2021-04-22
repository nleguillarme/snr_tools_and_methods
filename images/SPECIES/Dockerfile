FROM        ubuntu:xenial

# update and install dependencies
RUN         apt-get update \
                && apt-get install -y \
                    software-properties-common \
                    wget \
                && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
                && apt-get update \
                && apt-get install -y \
                    make \
                    curl \
                    vim \
                && apt-get install -y cmake=3.5.1-1ubuntu3 \
                && apt-get install -y \
                    gcc-4.9 g++-4.9 gcc-4.9-base \
                && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 100 \
                && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 100 \
                && apt-get install -y libboost-all-dev

RUN useradd species

WORKDIR /home/species

RUN wget http://download.jensenlab.org/species_tagger.tar.gz
RUN tar -xvf species_tagger.tar.gz

WORKDIR /home/species/species_tagger
RUN make species

ENV PATH="/home/species/species_tagger:${PATH}"
