FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装必要的软件包，包括SSH服务、客户端、sshpass和Python环境
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

# 创建普通用户
RUN useradd -m -s /bin/bash user && \
    echo 'user:123456' | chpasswd && \
    deluser user sudo || true && \
    deluser user adm || true

# 设置用户的 shell
RUN usermod -s /bin/bash user

RUN update-alternatives --set nc /bin/nc.traditional

# 暴露SSH服务端口
EXPOSE 22

# 启动SSH服务并保持运行
CMD ["/usr/sbin/sshd", "-D"]
