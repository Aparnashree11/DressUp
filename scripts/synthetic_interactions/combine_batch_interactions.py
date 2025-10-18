# Save as combine_batches_streaming.py
import json
from pathlib import Path
from tqdm import tqdm

def combine_batches_streaming(batch_dir, output_path):
    """Combine batches by streaming (memory efficient)"""
    print("[Combining] Streaming batches to output file...")
    
    batch_files = sorted(Path(batch_dir).glob("interactions_batch_*.json"))
    print(f"[Found] {len(batch_files)} batch files")
    
    total_interactions = 0
    
    with open(output_path, 'w') as outfile:
        outfile.write('[\n')  # Start JSON array
        
        first_batch = True
        
        for batch_file in tqdm(batch_files, desc="Combining"):
            with open(batch_file, 'r') as infile:
                batch_data = json.load(infile)
                
                for i, interaction in enumerate(batch_data):
                    # Add comma before each item except the first
                    if not first_batch or i > 0:
                        outfile.write(',\n')
                    
                    json.dump(interaction, outfile)
                    total_interactions += 1
                    first_batch = False
                
                # Clear memory
                del batch_data
        
        outfile.write('\n]')  # Close JSON array
    
    print(f"[Complete] {total_interactions:,} interactions combined")
    print(f"[Saved] {output_path}")

if __name__ == "__main__":
    combine_batches_streaming("./batches", "./synthetic_interactions.json")
