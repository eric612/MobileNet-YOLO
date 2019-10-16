# https://www.learnopencv.com/install-opencv3-on-ubuntu
# https://docs.opencv.org/3.4/d6/d15/tutorial_building_tegra_cuda.html

ARG CUDA_VERSION=10.1
ARG CUDNN_VERSION=7

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

ARG PYTHON_VERSION=2.7
ARG OPENCV_VERSION=4.1.1

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# ENV LD_LIBRARY_PATH /usr/local/${CUDA}/compat:$LD_LIBRARY_PATH

# COPY ./*.tar.gz /home/
# COPY caffe.zip /home
# RUN cd /home && \
#         tar -xvf cmake-3.14.7-Linux-x86_64.tar.gz && \
#         mv cmake-3.14.7-Linux-x86_64 /opt && \
#         echo "export PATH=/opt/cmake-3.14.7-Linux-x86_64/bin:$PATH" >> /etc/profile && \
#         . /etc/profile && \
#         cd .. 

# Add CUDA libs paths
#RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list \
RUN export DEBIAN_FRONTEND=noninteractive \
#         && apt-get clean && apt-get update && \
        && apt-get update && \
        CUDA_PATH=(/usr/local/cuda-*) && \
    CUDA=`basename $CUDA_PATH` && \
    echo "$CUDA_PATH/compat" >> /etc/ld.so.conf.d/${CUDA/./-}.conf && \
    ldconfig && \


# Install all dependencies for OpenCV and Caffe
    apt-get -y update --fix-missing && \
    apt-get -y install --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        $( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "python${PYTHON_VERSION%%.*}-distutils" ) \
        build-essential \
        wget \
        unzip \
        # cmake \
        git \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-scipy \
        python-skimage \
        libopencv-dev \
        libtbb2 \
        gfortran \
        apt-utils \
        pkg-config \
        checkinstall \
        qt5-default \
        build-essential \
        libatlas-base-dev \
        libglew-dev \
		python-opencv \
    && \
#     rm -rf /var/lib/apt/lists/* && \

# install python dependencies
    sysctl -w net.ipv4.ip_forward=1 && \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
#     pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \ 
    pip${PYTHON_VERSION} install numpy && \

# install CMake
    cd /home && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.7/cmake-3.14.7-Linux-x86_64.tar.gz && \
    tar -xvf cmake-3.14.7-Linux-x86_64.tar.gz && \
    mv cmake-3.14.7-Linux-x86_64 /opt && \
    echo "export PATH=/opt/cmake-3.14.7-Linux-x86_64/bin:$PATH" >> /etc/profile && \
    . /etc/profile && \

# # install nccl
#     tar -xvf nccl.tar.gz && \
#     cd /home/nccl && make -j install && cd .. && rm -rf nccl && \
#     rm nccl.tar.gz && \
    rm cmake-3.14.7-Linux-x86_64.tar.gz && \
    cd ..  && \


# Set the default python and install PIP packages
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \

# Call default command.
    python --version && \
    python -c "import cv2 ; print(cv2.__version__)"


# install Caffe

ENV PATH=/opt/cmake-3.14.7-Linux-x86_64/bin:$PATH



ENV MobileNetYOLO_ROOT=/workspace/MobileNet-YOLO
WORKDIR $MobileNetYOLO_ROOT

RUN cd /workspace && \
	git clone --depth 1 https://github.com/eric612/MobileNet-YOLO.git  && \
	#unzip caffe.zip && \
	cd $MobileNetYOLO_ROOT && \
    
    sed -i 's/set(Caffe_known_gpu_archs "20 21(20) 30 35 50 60 61")/set(Caffe_known_gpu_archs "30 35 50 60 61 75")/g' cmake/Cuda.cmake && \
    sed -i 's/set(__cuda_arch_bin "60 61")/set(__cuda_arch_bin "75")/g' cmake/Cuda.cmake && \
    sed -i 's/SET(CMAKE_C_COMPILER gcc-5)/# SET(CMAKE_C_COMPILER gcc-5)/g' CMakeLists.txt && \
    sed -i 's/SET(CMAKE_CXX_COMPILER g++-5)/# SET(CMAKE_CXX_COMPILER g++-5)/g' CMakeLists.txt && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 .. && \
    make -j"$(nproc)"   

ENV PYCAFFE_ROOT $MobileNetYOLO_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $MobileNetYOLO_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$MobileNetYOLO_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# WORKDIR /workspace

