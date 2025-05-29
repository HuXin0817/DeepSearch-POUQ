cd python_bindings
rm -rf build deepsearch.egg-info
pip uninstall deepsearch -y
pip install setuptools wheel numpy pybind11
pip install .
