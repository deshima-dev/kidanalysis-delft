#!/bin/bash
#NCPU=`python -c "import multiprocessing as m; print(m.cpu_count() - 1);"`

file_dir=$1
out_dir=$2

last_dir=$(basename "$file_dir")
second_last_dir=$(basename "$(dirname "$file_dir")")

#for file in ${file_dir}/PreadScan_*; do
#  echo -e "${file}/${out_dir}\ny" | python Configure.py
#  python FitSweep.py
#  python SaveFits.py
#done

export file_dir out_dir last_dir second_last_dir

cp ${file_dir}/kids.list /home/deshima/data/analysis/${second_last_dir}/${last_dir}/
parallel -j 12 --delay 5 '
  Preaddir=$(basename "{}")
  echo -e "{}\n/home/deshima/data/analysis/${second_last_dir}/${last_dir}/${Preaddir}/${out_dir}" | python Configure.py
  python FitSweep.py
  #python SaveFits.py
' ::: ${file_dir}/PreadScan_*

python AnaPowersweep.py /home/deshima/data/analysis/${second_last_dir}/${last_dir} ${out_dir}

#exit

kids_fullpath=/home/deshima/data/analysis/${second_last_dir}/${last_dir}/kids.list
timestamp=$(date +"%Y%m%d_%H%M%S")
destination_base_kids="/data/spacekids/data/ASTE2024/LT263_FlightChip/kidlist"
remote_machine="aste-d1c"
scp ${kids_fullpath} ${remote_machine}:${destination_base_kids}/${timestamp}_kids.list
ssh ${remote_machine} <<EOF
ln -sf ${destination_base_kids}/${timestamp}_kids.list /data/spacekids/data/ASTE2024/LT263_FlightChip/kids.list
EOF

echo ==== Symbolic link created   ln -sf ${destination_base_kids}/${timestamp}_kids.list /data/spacekids/data/ASTE2024/LT263_FlightChip/kids.list

ssh desql1 <<EOF
mkdir -p /home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/
EOF

scp /home/deshima/data/analysis/${second_last_dir}/${last_dir}/{fr_list_list.npy,P_list.npy,Freq_center_list_list.npy,kids.list} deshima@desql1:/home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/

ssh desql1 <<EOF
python plot_Psweep.py ${second_last_dir}/${last_dir}
EOF

