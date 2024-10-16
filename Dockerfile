FROM continuumio/miniconda3

COPY . /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN conda env create -f /app/environment.yml

RUN conda run -n FedCAP pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN echo "conda activate FedCAP" >> ~/.bashrc

WORKDIR /app

CMD ["bash"]
