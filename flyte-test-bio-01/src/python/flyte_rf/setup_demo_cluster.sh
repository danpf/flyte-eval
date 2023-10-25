#!/usr/bin/env bash

flytectl demo teardown -v

set -e
flytectl demo start

wait


DOCKER_OR_PODMAN=docker

export DOCKER_BUILDKIT=1

if [ "$DOCKER_OR_PODMAN" = "docker" ]; then
    x=proteinmpnn
    docker buildx build -f docker/Dockerfile.${x} --output type=image,name=localhost:30000/${x}:latest,push=true,compression=zstd,compression-level=1 .
    wait
    
    for x in biopython rfdiffusion esmfold foldingdiff;
    do
        docker buildx build -f docker/Dockerfile.${x} --output type=image,name=localhost:30000/${x}:latest,push=true,compression=zstd,compression-level=3 . &
    done
    wait
else
  x=proteinmpnn
  podman build . --format docker -f docker/Dockerfile.${x} -t ${x} && podman tag ${x} localhost:30000/${x}:latest && podman push --tls-verify=false localhost:30000/${x}:latest
  wait

  for x in biopython rfdiffusion esmfold foldingdiff;
  do
    wait
    x=proteinmpnn
    podman build . --format docker -f docker/Dockerfile.${x} -t ${x} && podman tag ${x} localhost:30000/${x}:latest && podman push --tls-verify=false localhost:30000/${x}:latest &
    wait
  done
fi

cat <<EOF > cra.yaml
attributes:
    projectQuotaCpu: "1000"
    projectQuotaMemory: 5Ti
project: flytesnacks
domain: development
EOF

cat <<EOF > tra.yaml
defaults:
    cpu: "2"
    memory: 1Gi
limits:
    cpu: "1000"
    memory: 5Ti
project: flytesnacks
domain: development
EOF

flytectl update task-resource-attribute --attrFile tra.yaml --config ~/.flyte/config-sandbox.yaml
wait
flytectl update cluster-resource-attribute --attrFile cra.yaml --config ~/.flyte/config-sandbox.yaml
wait

rm tra.yaml cra.yaml

set +e

