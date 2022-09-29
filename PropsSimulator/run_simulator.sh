cp Bulk_simulator.py Bulk_simulator.pyx
cp ST_simulator.py ST_simulator.pyx
python setup.py build_ext --inplace
python run_simulator.py
