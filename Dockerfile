# Start from a PyTorch base image with CUDA 10.2 and cuDNN 7
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Upgrade pip and install necessary Python packages
RUN pip install --upgrade pip
RUN pip install accelerate -U
RUN pip install torch==1.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the training script when the container launches
CMD ["python", "-m", "torch.distributed.launch", "--nproc_per_node=2", "train.py", "--arg1", "value1", "--arg2", "value2"]
