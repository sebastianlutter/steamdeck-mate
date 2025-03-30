#!/bin/bash
#
# Script to run and stop all stacks at once. When
# executed without parameter status is shown.
#
cd "$(dirname $0)"

title() {
  echo "##############################################################"
  echo "# $1"
  echo "##############################################################"
}

# find suitable docker-compose command
function find_compose_cmd() {
  podman compose ps &> /dev/null
  if [ $? -eq 0 ]; then
    echo "podman compose"
    return 0
  fi
  docker compose ps &> /dev/null
  if [ $? -eq 0 ]; then
    echo "docker compose"
    return 0
  fi
  podman-compose ps &> /dev/null
  if [ $? -eq 0 ]; then
    echo "podman-compose"
    return 0
  fi
  docker-compose ps &> /dev/null
  if [ $? -eq 0 ]; then
    echo "docker-compose"
    return 0
  fi
}

COMPOSE_CMD="$(find_compose_cmd)"
export LIST_ACTION="stats"
export NAME_FIELD=".Names"
echo $COMPOSE_CMD | grep docker &> /dev/null
if [ $? -ne 1 ]; then
  export LIST_ACTION="ls"
  export NAME_FIELD=".Name"
fi
if [ "$COMPOSE_CMD" != "" ]; then
  echo "Found \"$COMPOSE_CMD\" as docker compose command"
else
  echo "No docker compose or podman compose implementation found. Please install one."
  exit 1
fi

list_services() {
  $COMPOSE_CMD $LIST_ACTION
}

compose() {
  cd "$1"
  $COMPOSE_CMD -f ${CONF} $2 $3 $4
  cd ..
}

ACTION="${1}"
# set device suffix
case "${2}" in
  cuda|nvidia|gpu)
    echo "Starting NVIDIA cuda stacks"
    DEVICE="-cuda"
    ;;
  rocm|amd)
    echo "Starting ROCm stacks (not implemented yet)"
    DEVICE="-rocm"
    ;;
  cpu)
    DEVICE=""
    ;;
  *)
    # default for CPU
    DEVICE=""
    # probe for nvidia (auto detect)
    # Check if NVIDIA driver is installed and GPU is operational
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        echo "NVIDIA driver is installed and GPU is operational."
        if docker info | grep -q "Runtimes:.*nvidia"; then
            echo "NVIDIA container runtime is installed."
            echo "Using NVIDIA CUDA docker"
            DEVICE="-cuda"
        else
            echo "NVIDIA container runtime is not installed."
            echo "Starting CPU only stacks"
        fi
    elif command -v rocm-smi &> /dev/null && rocm-smi > /dev/null 2>&1; then
      echo "AMD rocm drivers are detected, using the amdgpu stack"
      DEVICE="-rocm"
    else
        echo "No GPU (NVIDIA or rocm) is installed or GPU is not operational."
        echo "Starting CPU only stacks"
    fi
    ;;
esac
CONF="docker-compose${DEVICE}.yml"
echo "Using $CONF files"

case "${ACTION}" in
  start)
    title "Starting all stacks"
    ;;
  stop)
    title "Stopping all stacks"
    ;;
  *)
    title "Show status"
    list_services
    ;;
esac

folder="./"

case "${ACTION}" in
  start)
     echo "Starting $folder"
     compose "$folder" up -d
    ;;
  stop)
     echo "Stop $folder"
     compose "$folder" down
    ;;
  *)
    title "Services of $folder"
    cd $folder
    $COMPOSE_CMD ps --format "{{$NAME_FIELD}} {{.Image}} {{.Status}}" | while read NAME IMAGE STATUS; do
      echo -e "  - $IMAGE - $STATUS: $NAME"
    done
    cd ..
    ;;
esac
