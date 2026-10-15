#!/usr/bin/env bash
# Three untouched runs in fresh processes (different PYTHONHASHSEED each, to expose hidden randomness).
# Compares the ENTIRE output file byte-for-byte, reason text included.
set -e
cd "$(dirname "$0")"
mkdir -p repro
for i in 1 2 3; do
  PYTHONHASHSEED=$((i*1000+7)) python3 run.py "${1:-/mnt/user-data/uploads/shipment_records.csv}" \
      "${2:-/mnt/user-data/uploads/context_notes.csv}" repro/run$i.csv >/dev/null
done
sha256sum repro/run1.csv repro/run2.csv repro/run3.csv | tee repro/hashes.txt
diff repro/run1.csv repro/run2.csv && diff repro/run2.csv repro/run3.csv \
  && echo "REPRODUCIBLE: 3 runs byte-identical (flags, numbers, cited notes AND reason text)" | tee -a repro/hashes.txt
