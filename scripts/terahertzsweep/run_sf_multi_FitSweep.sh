#!/bin/sh
NCPU=`python -c "import multiprocessing as m; print(m.cpu_count() - 1);"`

echo NCPU = $NCPU

for file in /Users/sfujita/Desktop/DESHIMA/toptica/widesweep_20240419_133537/run_20240419_133706/PreadScan_*; do
  echo "${file}/out20240422_1\ny" | python Configure.py
  python FitSweep.py
  #python FitSweep.py --mode plot --ncpu $NCPU
  python SaveFits.py
done