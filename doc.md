# ./ray run ./config
# export https_proxy=http://127.0.0.1:10808 
# export http_proxy=http://127.0.0.1:10808

# bash <(wget -qO- https://xuanyuan.cloud/docker.sh)
# sudo docker load -i my-python-env.tar
# sudo apt-get update
# sudo apt-get install -y nvidia-container-toolkit
# sudo systemctl restart docker

# sudo vim /etc/docker/daemon.json
``` {
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
# sudo docker run -it --rm --gpus all my-python-env:latest bash
# chmod +x ./xray 

