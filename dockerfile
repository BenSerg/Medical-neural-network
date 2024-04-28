FROM python:3.10
ARG TORCH_VERSION=2.2.1
ARG TORCHVISION_VERSION=0.17.1
ARG TORCHAUDIO_VERSION=2.2.1

ARG KEY

RUN if [ "$KEY" = "cuda" ]; then \
    echo "Running command for Ilya"; \
    pip3 install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION; \
elif [ "$KEY" = "rocm" ]; then \
    echo "Running command for Sergey"; \
    pip3 install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION --index-url https://download.pytorch.org/whl/rocm5.7; \
elif [ "$KEY" = "cpu" ]; then \
    echo "Running command for Mathew"; \
    pip3 install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION --index-url https://download.pytorch.org/whl/cpu; \
else \
    echo "Unknown key"; \
fi