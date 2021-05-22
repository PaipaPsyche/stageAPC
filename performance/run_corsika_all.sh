#!/bin/bash
cd /work/paipa/corsika-77400/run
for f in inputs/MONIIN*.txt
do
start_time="$(date -u +%s)"
./corsika77400Linux_QGSII_gheisha < $f
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds elapsed for $f" >> inputs/echotime.txt
# do something on $f
done