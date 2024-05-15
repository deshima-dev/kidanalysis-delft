#!/bin/sh
NCPU=`python -c "import multiprocessing as m; print(m.cpu_count() - 1);"`

echo NCPU = $NCPU

echo ====Configure.py====
python Configure.py


echo ====FitSweep.py====
python FitSweep.py
echo ====FitSweep.py --mode plot --ncpu $NCPU====
python FitSweep.py --mode plot --ncpu $NCPU


echo ====SaveFits.py====
python SaveFits.py
echo ====SaveFits.py --mode plot --ncpu $NCPU====
python SaveFits.py --mode plot --ncpu $NCPU

