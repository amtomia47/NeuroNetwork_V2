python -u "d:\NeuroNetwork_V2\WB\reload.py"
::python -u "d:\NeuroNetwork_V2\Main\compute_layer_output.py"
python -u "d:\NeuroNetwork_V2\Main\prediction_in_series.py"
python -u "d:\NeuroNetwork_V2\Prediction\error_rate.py" > untrained.txt
python -u "d:\NeuroNetwork_V2\Main\random_back_propagation_batch.py"
python -u "d:\NeuroNetwork_V2\Main\prediction_in_series.py"
python -u "d:\NeuroNetwork_V2\Prediction\error_rate.py" > trained.txt
type D:\NeuroNetwork_V2\untrained.txt
type D:\NeuroNetwork_V2\trained.txt
