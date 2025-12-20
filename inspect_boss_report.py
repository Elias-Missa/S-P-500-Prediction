
import pandas as pd
import os

report_path = r"c:\Users\eomis\SP500 Project\S-P-500-Prediction\ML_Output\ridge\Ridge_23rd_WalkForward_HUBER_12-19-2025_03-34\boss_report.xlsx"

try:
    print(f"Reading {report_path}...")
    xls = pd.ExcelFile(report_path)
    print("Sheets:", xls.sheet_names)
    
    sheet = 'Summary'
    print(f"\n--- {sheet} ---")
    df = pd.read_excel(report_path, sheet_name=sheet)
    print(df.to_string())
except Exception as e:
    print(f"Error: {e}")
