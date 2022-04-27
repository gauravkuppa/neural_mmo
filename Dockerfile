FROM ubuntu:20.04
ENV TZ=America/New_York

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends\
    python3.9 \
    python3-pip \
    git && \
    git clone https://github.com/gauravkuppa/neural_mmo.git && \
    python3.9 -m pip install --upgrade pip && \
    apt-get install --no-install-recommends -y python3.9-dev \
        build-essential \
        python3-venv \
        eog \
        python3-tk \
        python3-yaml \
        ssh \
        git

CMD [ "/bin/bash" ]