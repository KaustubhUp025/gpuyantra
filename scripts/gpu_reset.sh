#!/bin/bash
echo "Attempting GPU reset..."
sudo nvidia-smi --gpu-reset || sudo reboot
