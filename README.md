# @ scripts/terahertzsweep
These shell scripts are for Toptica THz sweep analysis and Readout power sweep analysis
## run_sf.sh (for Toptica THz sweep analysis)

ex.)

```
$ ./run_sf.sh /home/deshima/data/LT263_FlightChip/run_20240421_004647/TerahertzScan_20240421_005117/ out_test
```

```shell
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

echo ====Symbolic link created   ln -sf ${destination_base_json}/${timestamp}_kid_corresp.json /data/spacekids/data/ASTE2024/LT263_FlightChip/kid_corresp.json

ssh desql1 <<EOF
mkdir -p /home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/${out_dir}
EOF

scp /home/deshima/data/analysis/${second_last_dir}/${last_dir}/${out_dir}/{reference.dat,reference.png,kid_corresp.json} deshima@desql1:/home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/${out_dir}/
scp /home/deshima/data/analysis/${second_last_dir}/${last_dir}/${out_dir}/KIDCorresp/*png deshima@desql1:/home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/${out_dir}/


echo ====FINISHED====
```

This is a shell script that executes the following python scripts.

Comment out and run it as appropriate.

The default is to use a maximum of -1 CPU.

* Configure.py
  * Specify a new directory in which to place the analysis results. If the directory already exists, this will fail.
* FitSweep.py
  \*
* SaveFits.py
  \*
* THzFrequencyTOD.py
  * This generates reference.dat, which lists the KIDs that are performing worse than a certain threshold. The threshold value is specified with --refvalue. The higher the threshold value, the fewer KIDs will be listed in reference.dat.
* AnaSpectrum.py
  \*
* KIDCorresp.py
  * At the top of the script, specify the appropriate kid_test.db path. Also, edit the "detector_version =" line appropriately.

Later in the script, "kid_corresp.json is transferred to aste-d1c and a symbolic link will be created.

## run_powersweep.sh (for Readout power sweep analysis)

ex.)

```
$ ./run_powersweep.sh ~/data/LT263_FlightChip/widesweep_20240423_140629/run_20240423_140758/ out_test
```

```shell
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

echo ====Symbolic link created   ln -sf ${destination_base_kids}/${timestamp}_kids.list /data/spacekids/data/ASTE2024/LT263_FlightChip/kids.list

ssh desql1 <<EOF
mkdir -p /home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/
EOF

scp /home/deshima/data/analysis/${second_last_dir}/${last_dir}/{fr_list_list.npy,P_list.npy,Freq_center_list_list.npy,kids.list} deshima@desql1:/home/deshima/data/fujita_analysis/${second_last_dir}/${last_dir}/

ssh desql1 <<EOF
python plot_Psweep.py ${second_last_dir}/${last_dir}
EOF



```

This script runs FitSweep.py on multiple data (PreadScan\_\*).

In this script, AnaPowersweep.py has been included.

This script finds the appropriate read power and writes them (-4 dBm) to "kids.list".

The original "kids.list" is renamed with a timestamp.

Later in the script, "kids.list" is transferred to aste-d1c and a symbolic link will be created.


# @ scripts/aste
This shell script is for both Noise analysis and COSMOS data analysis
## run_sf.sh

ex.)

```
$ ./run_sf.sh /home/deshima/data/LT263_FlightChip/run_20240423_145620/ out_test
```

```shell
#!/bin/sh
NCPU=`python -c "import multiprocessing as m; print(m.cpu_count() - 1);"`

echo NCPU = $NCPU

file_dir=$1
out_dir=$2

last_dir=$(basename "$file_dir")
#second_last_dir=$(basename "$(dirname "$file_dir")")

echo ====Configure.py====
echo -e "${file_dir}\n/home/deshima/data/analysis/${last_dir}/${out_dir}" | python Configure.py


echo ====FitSweep.py====
python FitSweep.py
echo ====FitSweep.py --mode plot --ncpu $NCPU====
python FitSweep.py --mode plot --ncpu $NCPU


echo ====SaveFits.py====
python SaveFits.py
echo ====SaveFits.py --mode plot --ncpu $NCPU====
python SaveFits.py --mode plot --ncpu $NCPU


echo ====FINISHED====
```

This is a shell script that executes the following python scripts.

Comment out and run it as appropriate.

The default is to use a maximum of -1 CPU.

* Configure.py
  * Specify a new directory in which to place the analysis results. If the directory already exists, this will fail.
* FitSweep.py
  \*
* SaveFits.py
  * Creates reduced fits file

