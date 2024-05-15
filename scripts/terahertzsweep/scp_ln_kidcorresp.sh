#!/bin/bash

file_dir=$1

json_fullpath=${file_dir}/kid_corresp.json
timestamp=$(date +"%Y%m%d_%H%M%S")
destination_base_json="/data/spacekids/data/ASTE2024/LT263_FlightChip/kidcorresp"
remote_machine="aste-d1c"
scp ${json_fullpath} ${remote_machine}:${destination_base_json}/${timestamp}_kid_corresp.json
ssh ${remote_machine} <<EOF
ln -sf ${destination_base_json}/${timestamp}_kid_corresp.json /data/spacekids/data/ASTE2024/LT263_FlightChip/kid_corresp.json
EOF
