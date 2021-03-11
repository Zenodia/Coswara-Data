#!/bin/bash
DOCKER_IMAGE="triton_pytorch_cuda11"
CONTAINER_NAME="triton_pytorch_cuda11"


sudo docker build -t triton_pytorch_cuda11 ./Docker/



GPU_IDs=$1
DATA_DIR=$2
PWD=`pwd`

echo -----------------------------------
echo starting docker for ${DOCKER_IMAGE} using GPUS ${GPU_IDs}
echo -----------------------------------

#################################### check if name is used then exit
sudo docker ps|grep ${CONTAINER_NAME}
dockerNameExist=$?
if ((${dockerNameExist}==0)) ;then
  echo --- dockerName ${CONTAINER_NAME} already exist
  echo ----------- attaching into the docker
  docker exec -it ${CONTAINER_NAME} /bin/bash
  exit
fi

extraFlag="-it --ipc=host --net=host"
cmd2run="/bin/bash"

echo starting please run "./installDashBoardInDocker.sh" to install the lab extensions then start the jupeter lab
echo once completed use web browser with token given yourip:${jnotebookPort} to access it

echo $DOCKER_IMAGE

sudo docker run ${extraFlag} \
  -it --rm -p 23579:23579  \
  --name=${CONTAINER_NAME} \
  --gpus "device="${GPU_IDs} \
  -v ${PWD}/:/workspace/codes/ \
  -v ${DATA_DIR}:/workspace/data \
  -w /workspace/codes \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  ${DOCKER_IMAGE} \
  ${cmd2run}
