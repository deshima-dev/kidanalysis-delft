#!/bin/sh
NCPU=`python -c "import multiprocessing as m; print(m.cpu_count() - 1);"`

echo NCPU = $NCPU

file_dir=$1
out_dir=$2

last_dir=$(basename "$file_dir")
second_last_dir=$(basename "$(dirname "$file_dir")")

echo ====Configure.py====
echo -e "${file_dir}\n/home/deshima/data/analysis/${second_last_dir}/${last_dir}/${out_dir}" | python Configure.py

echo ====FitSweep.py====
python FitSweep.py
echo ====FitSweep.py --mode plot --ncpu $NCPU====
python FitSweep.py --mode plot --ncpu $NCPU


echo ====SaveFits.py====
python SaveFits.py
echo ====SaveFits.py --mode plot --ncpu $NCPU====
python SaveFits.py --mode plot --ncpu $NCPU


echo ====THzFrequencyTOD.py --refvalue 5.0====
python THzFrequencyTOD.py --refvalue 5.0
echo ====THzFrequencyTOD.py --mode plot --ncpu $NCPU====
python THzFrequencyTOD.py --mode plot --ncpu $NCPU


echo ====python AnaSpectrum.py --mode 1 --ncpu $NCPU====
###python AnaSpectrum.py --mode 1 --ncpu $NCPU
echo ====python AnaSpectrum.py --mode 2====
python AnaSpectrum.py --mode 2


echo ====KIDCorresp.py====
python KIDCorresp.py

json_fullpath=/home/deshima/data/analysis/${second_last_dir}/${last_dir}/${out_dir}/kid_corresp.json
timestamp=$(date +"%Y%m%d_%H%M%S")
destination_base_json="/data/spacekids/data/ASTE2024/LT263_FlightChip/kidcorresp"
remote_machine="aste-d1c"
scp ${json_fullpath} ${remote_machine}:${destination_base_json}/${timestamp}_kid_corresp.json
ssh ${remote_machine} <<EOF
ln -sf ${destination_base_json}/${timestamp}_kid_corresp.json /data/spacekids/data/ASTE2024/LT263_FlightChip/kid_corresp.json
EOF

echo ==== Symbolic link created => ln -sf ${destination_base_json}/${timestamp}_kid_corresp.json /data/spacekids/data/ASTE2024/LT263_FlightChip/kid_corresp.json

ssh desql1 <<EOF
mkdir -p /home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/${out_dir}
EOF

scp /home/deshima/data/analysis/${second_last_dir}/${last_dir}/${out_dir}/{reference.dat,reference.png,kid_corresp.json} deshima@desql1:/home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/${out_dir}/
scp /home/deshima/data/analysis/${second_last_dir}/${last_dir}/${out_dir}/KIDCorresp/*png deshima@desql1:/home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/${out_dir}/


echo ====FINISHED====