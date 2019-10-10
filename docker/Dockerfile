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
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list \
        && export DEBIAN_FRONTEND=noninteractive \
        && apt-get clean && apt-get update && \
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
        # libopencv-dev \
        libtbb2 \
        gfortran \
        apt-utils \
        pkg-config \
        checkinstall \
        qt5-default \
        build-essential \
        libatlas-base-dev \
        libgtk2.0-dev \
        libavcodec57 \
        libavcodec-dev \
        libavformat57 \
        libavformat-dev \
        libavutil-dev \
        libswscale4 \
        libswscale-dev \
        libjpeg8-dev \
        libpng-dev \
        libtiff5-dev \
        libdc1394-22 \
        libdc1394-22-dev \
        libxine2-dev \
        libv4l-dev \
        libgstreamer1.0 \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-base1.0-dev \
        libglew-dev \
        libpostproc-dev \
        libeigen3-dev \
        libtbb-dev \
        zlib1g-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
    && \
#     rm -rf /var/lib/apt/lists/* && \

# install python dependencies
    sysctl -w net.ipv4.ip_forward=1 && \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \ 
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

# Install OpenCV
    cd /home && \
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip -O opencv.zip --progress=bar:force:noscroll && \
    unzip -q opencv.zip && \

    # tar -xvf opencv-4.1.1.tar.gz && \
    mv opencv-$OPENCV_VERSION /opencv && \
    rm opencv.zip && \
    # rm opencv-4.1.1.tar.gz && \
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip -O opencv_contrib.zip --progress=bar:force:noscroll && \
    unzip -q opencv_contrib.zip && \
    # tar -xvf opencv_contrib-4.1.1.tar.gz && \
    mv opencv_contrib-$OPENCV_VERSION /opencv_contrib && \
    rm opencv_contrib.zip && \
    # rm opencv_contrib-4.1.1.tar.gz && \
    cd ..  && \

# Prepare build
    mkdir /opencv/build && \
    cd /opencv/build && \
    cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_PYTHON_SUPPORT=ON \
      -D BUILD_DOCS=ON \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_TESTS=OFF \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
      -D BUILD_opencv_python3=$( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "ON" || echo "OFF" ) \
      -D BUILD_opencv_python2=$( [ ${PYTHON_VERSION%%.*} -lt 3 ] && echo "ON" || echo "OFF" ) \
      -D PYTHON${PYTHON_VERSION%%.*}_EXECUTABLE=$(which python${PYTHON_VERSION}) \
      -D PYTHON_DEFAULT_EXECUTABLE=$(which python${PYTHON_VERSION}) \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_IPP=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_TBB=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D WITH_CUDA=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
      -D CUDA_ARCH_PTX="" \
      -D WITH_CUBLAS=ON \
      -D WITH_NVCUVID=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      .. \
    && \

# Build, Test and Install
    cd /opencv/build && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \

# cleaning
   apt-get -y remove \
#         unzip \
#         cmake \
        gfortran \
#         apt-utils \
#         pkg-config \
#         checkinstall \
#         build-essential \
#         libatlas-base-dev \
#         libgtk2.0-dev \
#         libavcodec-dev \
#         libavformat-dev \
#         libavutil-dev \
#         libswscale-dev \
#         libjpeg8-dev \
#         libpng12-dev \
#         libtiff5-dev \
#         libdc1394-22-dev \
#         libxine2-dev \
#         libv4l-dev \
#         libgstreamer1.0-dev \
#         libgstreamer-plugins-base1.0-dev \
#         libglew-dev \
#         libpostproc-dev \
#         libeigen3-dev \
#         libtbb-dev \
#         zlib1g-dev \
    && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /opencv /opencv_contrib /var/lib/apt/lists/* && \

# Set the default python and install PIP packages
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \

# Call default command.
    python --version && \
    python -c "import cv2 ; print(cv2.__version__)"


# install Caffe

ENV PATH=/opt/cmake-3.14.7-Linux-x86_64/bin:$PATH

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=master


RUN echo `pwd` && \ 
    ldconfig && \
    cd /home && \
    git clone -b ${CLONE_TAG} --depth 1 https://github.com/intel/caffe.git .  && \
# RUN  echo `pwd` && \
    unzip caffe.zip && \
#     tar -xvf nccl.tar.gz && \
    mv caffe /opt && \
    cd $CAFFE_ROOT && \
#     echo `pwd` && \
#     pip install --upgrade pip && \
#     sed -i 's/from pip import main/from pip import __main__/g' /usr/bin/pip && \
#     sed -i 's/sys.exit(main())/sys.exit(__main__._main())/g' /usr/bin/pip && \
#     pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \ 
    sed -i 's/python-dateutil>=1.4,<2/python-dateutil/g' python/requirements.txt && \
    sed -i 's/networkx>=1.8.1/networkx==2.2/g' python/requirements.txt && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    sed -i 's/set(Caffe_known_gpu_archs "20 21(20) 30 35 50 60 61")/set(Caffe_known_gpu_archs "30 35 50 60 61 75")/g' cmake/Cuda.cmake && \
    sed -i 's/set(__cuda_arch_bin "60 61")/set(__cuda_arch_bin "75")/g' cmake/Cuda.cmake && \
    sed -i 's/cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);/cv_img = cv::imread(image.first, cv::IMREAD_COLOR);/g' src/caffe/layers/window_data_layer.cpp && \
    sed -i 's/out_img = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);/out_img = cv::imdecode(buf, cv::IMREAD_COLOR);/g' src/caffe/layers/window_data_layer.cpp && \
    sed -i 's/CV_LOAD_IMAGE_GRAYSCALE);/cv::IMREAD_GRAYSCALE);/g' src/caffe/util/io.cpp && \
    sed -i 's/int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :/int cv_read_flag = (is_color ? cv::IMREAD_COLOR :/g' src/caffe/util/io.cpp && \
    sed -i 's/int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :/int cv_read_flag = (is_color ? cv::IMREAD_COLOR :/g' src/caffe/test/test_io.cpp && \
    sed -i 's/CV_LOAD_IMAGE_GRAYSCALE);/cv::IMREAD_GRAYSCALE);/g' src/caffe/test/test_io.cpp && \
    sed -i 's/caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)/caffe_option(USE_NCCL "Build Caffe with NCCL library support" ON)/g'  CMakeLists.txt && \
    sed -i 's/caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)/caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" ON)/g' CMakeLists.txt && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
#     cd /home/nccl && make -j install && cd .. && rm -rf nccl && cd $CAFFE_ROOT && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 .. && \
    make -j"$(nproc)"

ENV MobileNetYOLO_ROOT=/MobileNet-YOLO
WORKDIR $MobileNetYOLO_ROOT

RUN cd / && \
    git clone https://github.com/eric612/MobileNet-YOLO.git  && \
    sed -i 's/set(Caffe_known_gpu_archs "20 21(20) 30 35 50 60 61")/set(Caffe_known_gpu_archs "30 35 50 60 61 75")/g' cmake/Cuda.cmake && \
    sed -i 's/set(__cuda_arch_bin "60 61")/set(__cuda_arch_bin "75")/g' cmake/Cuda.cmake && \ 
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 .. && \
    make -j"$(nproc)"   

ENV PYCAFFE_ROOT $MobileNetYOLO_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $MobileNetYOLO_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$MobileNetYOLO_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# WORKDIR /workspace

