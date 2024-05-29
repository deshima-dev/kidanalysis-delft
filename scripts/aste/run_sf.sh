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
