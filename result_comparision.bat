@ echo off
for /l %%x in (1,1,10) do (
    python -u "d:\NeuroNetwork_V2\WB\reload.py"
    python -u "d:\NeuroNetwork_V2\Main\compute_final_output.py"
    python -u "d:\NeuroNetwork_V2\Main\random_back_propagation_batch.py"
    python -u "d:\NeuroNetwork_V2\Main\compute_final_output.py"
    python -u "D:\NeuroNetwork_V2\Backup\save_trained.py"
    python -u "D:\NeuroNetwork_V2\Outcome_analyze\trained_analyze.py"
)
