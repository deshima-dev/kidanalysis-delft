import os
import re
from glob import glob
import time

start = time.time()

candidates = glob("/home/deshima/data/LT263_FlightChip/run_2024*/TerahertzScan_*")

#pattern = re.compile(r"/run_20241[0-2][0-3][0-9]_\d{6}/")
pattern = re.compile(r"/run_2024(0[7-9]|1[0-2])[0-3][0-9]_\d{6}/")

data_list = sorted([path for path in candidates if pattern.search(path)])
#print(data_list)

count = 0
for data in data_list:
    print("#######")
    print("#######")
    print("#######")
    print(data)
    try:
        os.system("./run_sf_2025.sh %s out_20250528_3"%data)
        os.system("rm /home/deshima/data/fujita_analysis/ana_2025/analysis/run_2024*/TerahertzScan_*/out_20250528_3/reduced_measurement.fits")
    except:
        print("ERROR")
        print(data)
    count += 1
    print("count = ", count)

end = time.time()
print("#######")
print("#######")
print("#######")
print("TotalTime: ", end - start)
print("count = ", count)
