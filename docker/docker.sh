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

list_services() {
  docker compose ls
}

compose() {
  cd "$1"
  docker compose -f ${CONF} $2 $3 $4
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
    else
        echo "NVIDIA driver is not installed or GPU is not operational."
        echo "Starting CPU only stacks"
    fi
    ;;
esac
CONF="docker-compose${DEVICE}.yaml"
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
    docker compose ps --format "{{.Name}} {{.Image}} {{.Status}}" | while read NAME IMAGE STATUS; do
      echo -e "  - $IMAGE - $STATUS: $NAME"
    done
    cd ..
    ;;
esac
