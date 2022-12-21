# Start from a azure ml inference curated base image
FROM mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest

USER root

RUN apt-get update \
    && apt-get install -y curl jq 

USER dockeruser

# copy the conda.yml inside the container
COPY conda.yml /tmp/conda.yml

# azure ml curated image has dependencies installed in a conda env called amlenv. Lets add our dependencies to the same env.
RUN conda env update -n amlenv --file /tmp/conda.yml
