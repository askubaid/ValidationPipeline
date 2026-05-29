import torch
import os
import sys
from comet import download_model, load_from_checkpoint

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import config

def main():
    # Set the cache directory for huggingface to save models locally in the project dir
    os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'model_cache')
    
    src_base, src_ext = os.path.splitext(config.SRC_FILE_NAME)
    tgt_base, tgt_ext = os.path.splitext(config.TGT_FILE_NAME)
    
    # Layer 2 reads the clean output from Layer 1
    en_file = os.path.join(PROJECT_ROOT, "layer1", f"{src_base}_clean{src_ext}")
    pt_file = os.path.join(PROJECT_ROOT, "layer1", f"{tgt_base}_clean{tgt_ext}")
    
    layer2_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Checking for GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "Unbabel/wmt22-cometkiwi-da"
    saving_dir = "comet_models"
    
    print("Checking for local COMET-QE model...")
    model_path = None
    if os.path.exists(saving_dir):
        for root, dirs, files in os.walk(saving_dir):
            for file in files:
                if file.endswith(".ckpt") and "wmt22-cometkiwi-da" in root:
                    model_path = os.path.join(root, file)
                    break
            if model_path:
                break

    if model_path:
        print(f"Loading local model from {model_path}...")
    else:
        print("Downloading COMET-QE model from Hugging Face...")
        model_path = download_model(model_name, saving_directory=saving_dir)
        
    model = load_from_checkpoint(model_path)

    data = []
    print(f"Reading {en_file} and {pt_file}...")
    with open(en_file, "r", encoding="utf-8") as f_en, \
         open(pt_file, "r", encoding="utf-8") as f_pt:
        for idx, (en_line, pt_line) in enumerate(zip(f_en, f_pt)):
            data.append({
                "id": idx,
                "src": en_line.strip(),
                "tgt": pt_line.strip(),
            })
            
    # Prepare data for COMET. COMET-QE expects list of dicts: {"src": "...", "mt": "..."}
    comet_data = [{"src": row["src"], "mt": row["tgt"]} for row in data]
    
    print(f"Scoring {len(comet_data)} sentences... This may take a while depending on GPU.")
    # The predict method handles batching automatically.
    # We set batch_size=16 which should comfortably fit in an RTX 2080 8GB VRAM.
    model_output = model.predict(comet_data, batch_size=16, gpus=1 if device=="cuda" else 0)
    
    # model_output contains .scores (list of scores for each segment) and .system_score
    scores = model_output.scores

    print("Writing results...")
    
    excellent_en = os.path.join(layer2_dir, f"{src_base}_excellent{src_ext}")
    excellent_pt = os.path.join(layer2_dir, f"{tgt_base}_excellent{tgt_ext}")
    good_en = os.path.join(layer2_dir, f"{src_base}_good{src_ext}")
    good_pt = os.path.join(layer2_dir, f"{tgt_base}_good{tgt_ext}")
    mediocre_en = os.path.join(layer2_dir, f"{src_base}_mediocre{src_ext}")
    mediocre_pt = os.path.join(layer2_dir, f"{tgt_base}_mediocre{tgt_ext}")
    
    with open(excellent_en, "w", encoding="utf-8") as f_exc_en, \
         open(excellent_pt, "w", encoding="utf-8") as f_exc_pt, \
         open(good_en, "w", encoding="utf-8") as f_good_en, \
         open(good_pt, "w", encoding="utf-8") as f_good_pt, \
         open(mediocre_en, "w", encoding="utf-8") as f_med_en, \
         open(mediocre_pt, "w", encoding="utf-8") as f_med_pt:
         
         for row, score in zip(data, scores):
             if score >= 0.8:
                 f_exc_en.write(row["src"] + "\n")
                 f_exc_pt.write(row["tgt"] + "\n")
             elif score >= 0.7:
                 f_good_en.write(row["src"] + "\n")
                 f_good_pt.write(row["tgt"] + "\n")
             else:
                 f_med_en.write(row["src"] + "\n")
                 f_med_pt.write(row["tgt"] + "\n")
                 
    print(f"Layer 2 complete. Categorized files saved in {layer2_dir}.")

if __name__ == "__main__":
    main()
