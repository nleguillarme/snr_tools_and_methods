FROM openjdk:8

# update and install dependencies
RUN         apt-get update \
                && apt-get install -y \
                    software-properties-common \
                    wget \
                    unzip

RUN useradd linnaeus

WORKDIR /home/linnaeus

# Download LINNAEUS
RUN wget https://downloads.sourceforge.net/project/linnaeus/Linnaeus/linnaeus-2.0.tar.gz \
   && tar -xvf linnaeus-2.0.tar.gz

# Download dictionary
RUN wget https://downloads.sourceforge.net/project/linnaeus/Entity_packs/species-proxy-1.2.tar.gz \
   && tar -xvf species-proxy-1.2.tar.gz

RUN sed -i '2s#.*#$dir = /home/linnaeus/species-proxy/#' /home/linnaeus/species-proxy/properties.conf

RUN mkdir /home/linnaeus/corpus


