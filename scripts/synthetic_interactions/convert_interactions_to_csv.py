"""
Convert large JSON to CSV for memory-efficient loading
"""

import json
import csv
from tqdm import tqdm

def json_to_csv(json_path, csv_path, max_rows=None):
    """Convert JSON to CSV efficiently"""
    
    print(f"[Converting] {json_path} -> {csv_path}")
    
    with open(json_path, 'r') as f_in, open(csv_path, 'w', newline='') as f_out:
        # Skip opening bracket
        f_in.readline()
        
        writer = None
        count = 0
        
        for line in tqdm(f_in, desc="Converting"):
            if max_rows and count >= max_rows:
                break
            
            line = line.strip()
            if line and line not in [']', '[', ',']:
                if line.endswith(','):
                    line = line[:-1]
                
                try:
                    data = json.loads(line)
                    
                    # Initialize CSV writer with headers
                    if writer is None:
                        headers = list(data.keys())
                        writer = csv.DictWriter(f_out, fieldnames=headers)
                        writer.writeheader()
                    
                    writer.writerow(data)
                    count += 1
                    
                except Exception as e:
                    continue
    
    print(f"[Done] Converted {count:,} rows to CSV")
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    json_to_csv(
        "synthetic_interactions.json",
        "interactions.csv",
        max_rows=1000000  # 1M cap for training
    )
