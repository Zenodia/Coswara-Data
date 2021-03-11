sudo docker run --runtime=nvidia --gpus '"device=0"' -it --rm -v $(pwd):/workspace --shm-size=8g \
-p $1:$1 -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:1.0.0b3
