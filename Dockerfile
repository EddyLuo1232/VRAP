FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ufw \
    iptables \
    socat \
    steghide \
    imagemagick \
    php \
    hydra \
    udev \
    telnet \
    knockd \
    ldb-tools \
    sudo \
    bash \
    openssl \
    openssh-server \
    openssh-client \
    sshpass \
    python3 \
    python3-pip \
    nodejs \
    npm \
    net-tools \
    dnsutils \
    traceroute \
    iputils-ping \
    curl \
    wget \
    netcat-traditional \
    whois \
    nmap \
    iproute2 \
    nfs-common \
    rpcbind \
    && mkdir -p /var/run/sshd \
    && mkdir -p /root/.ssh && chmod 700 /root/.ssh \
    && touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys \
    && echo 'root:ubuntu' | chpasswd \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/openwall/john.git && \
    cd john/src && \
    ./configure && \
    make -s clean && \
    make -j2

RUN echo "alias john='/john/run/john'" >> /root/.bashrc

RUN useradd -m -s /bin/bash user && \
    echo 'user:123456' | chpasswd && \
    deluser user sudo || true && \
    deluser user adm || true

RUN usermod -s /bin/bash user

RUN update-alternatives --set nc /bin/nc.traditional

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
