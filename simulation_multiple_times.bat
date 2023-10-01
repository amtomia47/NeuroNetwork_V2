@echo off

for /l %%x in (1, 1, 100) do (
   cmd /c "D:\NeuroNetwork_V2\simulation_reload_and_restart.bat"
   python -u "D:\NeuroNetwork_V2\Outcome_analyze\trained_result_record.py"
)