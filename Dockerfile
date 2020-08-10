FROM ubuntu:20.04

RUN apt update && apt upgrade -y

# do it like this, otherwise a prompt for timezone will be there
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

# install build tools, etc.
RUN apt install -y cmake build-essential

# finally install opencv (opencv-dev seems to be needed for cmake detection of the lib)
RUN apt install -y libopencv-core4.2 libopencv-dev