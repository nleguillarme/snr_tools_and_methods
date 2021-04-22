FROM        ubuntu:xenial

# update and install dependencies
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
RUN curl --silent --location https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs
RUN apt-get install -y build-essential
RUN apt-get install -y npm

RUN useradd taxonfinder

WORKDIR /home/taxonfinder

RUN npm install taxonfinder --save
RUN npm install glob --save

COPY taxonfinder.js /home/taxonfinder/taxonfinder.js
