FROM ubuntu:18.04

COPY mcdeploy.key /mcdeploy.key
ARG current_branch

RUN apt-get -y update
RUN apt-get install -y \
        cmake \
	curl \
        g++ \
        clang-format \
        git \
        htop \
        libboost-all-dev \
        libeigen3-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        make \
        python3-dev \
        python3-pip \
        zlib1g-dev \
        ;

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash
RUN apt-get -y install nodejs

RUN pip3 install torch


# Install Git LFS
COPY gitlfs_install.sh /gitlfs_install.sh
RUN apt-get install -y software-properties-common  # for add-apt-repository
RUN add-apt-repository ppa:git-core/ppa \
        && bash /gitlfs_install.sh \
        && apt-get install -y git-lfs \
        && git lfs install


# Clone/make repo
ENV GIT_SSH_COMMAND "ssh -i /mcdeploy.key -o StrictHostKeyChecking=no"
RUN git lfs clone --recursive git@github.com:fairinternal/minecraft.git
WORKDIR minecraft
RUN git checkout ${current_branch}
RUN curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/models_folder.tar.gz -o models_folder.tar.gz
RUN tar -xzvf models_folder.tar.gz -C python/craftassist/models/ --strip-components 1
RUN make

RUN pip3 install -r requirements.txt

RUN mkdir workdir
WORKDIR workdir

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY run.fromenv.sh /run.fromenv.sh

EXPOSE 3000
EXPOSE 2556
EXPOSE 2557
EXPOSE 25565 
EXPOSE 9000
EXPOSE 5000

ENTRYPOINT ["/run.fromenv.sh"]  # uses $RUN_SH_GZ_B64
