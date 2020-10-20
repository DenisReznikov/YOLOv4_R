FROM python:3.6

COPY requirements.txt .
ENV HOME /root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
	locales \
        wget \
        software-properties-common \
        ca-certificates \
        build-essential \
        cmake \
        git \
        libopencv-dev \
        python3-dev \
        python3-pip \
        libgtk2.0-dev\
        pkg-config\
    && apt-get -y autoremove \
    && apt-get clean \
    && pip3 install setuptools wheel \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN pip install -r requirements.txt


WORKDIR ${HOME}
#Checkout version should be 4.4.0 when the new version is released 
RUN git clone http://github.com/opencv/opencv.git && cd opencv \
    && git checkout 4.4.0    \
    && mkdir build && cd build              \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE    \
        -D CMAKE_INSTALL_PREFIX=/usr/local  \
        -D WITH_CUDA=OFF                     \
        -D WITH_OPENCL=OFF                  \
        -D ENABLE_FAST_MATH=1               \
        -D CUDA_FAST_MATH=1                 \
        -D WITH_CUBLAS=1                    \
        -D BUILD_DOCS=OFF                   \
        -D BUILD_PERF_TESTS=OFF             \
        -D BUILD_TESTS=OFF                  \
        ..                                  \
    && make -j `nproc`                      \
    && make install                         \
    && cd ${HOME} && rm -rf ./opencv/


RUN git clone https://github.com/DenisReznikov/YOLOv4_R.git && cd YOLOv4_R \
    && make \
    && wget https://github.com/DenisReznikov/YOLOv4_R/releases/download/v1.0/custom-yolov4-detector_best.weights

WORKDIR /root/YOLOv4_R


CMD ["python", "test.py"]




