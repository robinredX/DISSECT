cp Bulk_simulator.py Bulk_simulator.pyx
cp ST_simulator.py ST_simulator.pyx
python3 setup.py build_ext --inplace
python3 run_simulator.py
