import os
import sys
import re
import csv
import urllib.request
import fasttext
import shutil

# Add the project root to the python path so we can import config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import config

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_PATH = os.path.join(PROJECT_ROOT, "lid.176.bin")

def download_fasttext_model():
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"Downloading fasttext model to {FASTTEXT_MODEL_PATH}...")
        try:
            urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download fasttext model: {e}")
            print("Please download it manually from:", FASTTEXT_MODEL_URL)
            exit(1)

def has_html_or_markdown(text):
    if re.search(r'<[^>]+>', text):
        return True
    if re.search(r'\[.*?\]\(.*?\)', text) or re.search(r'\*\*.*?\*\*', text) or re.search(r'\*.*?\*', text):
        return True
    return False

def has_control_chars(text):
    return bool(re.search(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', text))

def main():
    download_fasttext_model()
    print("Loading language identification model...")
    model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    en_file = os.path.join(PROJECT_ROOT, config.SRC_FILE_NAME)
    pt_file = os.path.join(PROJECT_ROOT, config.TGT_FILE_NAME)
    
    import contextlib

    layer1_dir = os.path.dirname(os.path.abspath(__file__))
    
    src_base, src_ext = os.path.splitext(config.SRC_FILE_NAME)
    tgt_base, tgt_ext = os.path.splitext(config.TGT_FILE_NAME)
    
    clean_en_file = os.path.join(layer1_dir, f"{src_base}_clean{src_ext}")
    clean_pt_file = os.path.join(layer1_dir, f"{tgt_base}_clean{tgt_ext}")
    output_csv = os.path.join(layer1_dir, "layer1_results.csv")
    
    categories = ["length_ratio", "markdown", "identical", "LangID", "control_char", "duplicates"]

    print(f"Processing {en_file} and {pt_file}...")
    
    flagged_count = 0
    clean_count = 0
    seen_src = set()
    
    with contextlib.ExitStack() as stack:
        f_en = stack.enter_context(open(en_file, "r", encoding="utf-8"))
        f_pt = stack.enter_context(open(pt_file, "r", encoding="utf-8"))
        f_clean_en = stack.enter_context(open(clean_en_file, "w", encoding="utf-8"))
        f_clean_pt = stack.enter_context(open(clean_pt_file, "w", encoding="utf-8"))
        f_out = stack.enter_context(open(output_csv, "w", encoding="utf-8", newline=""))
        
        cat_files = {}
        for cat in categories:
            cat_en_path = os.path.join(layer1_dir, f"{src_base}_{cat}{src_ext}")
            cat_pt_path = os.path.join(layer1_dir, f"{tgt_base}_{cat}{tgt_ext}")
            cat_files[cat] = (
                stack.enter_context(open(cat_en_path, "w", encoding="utf-8")),
                stack.enter_context(open(cat_pt_path, "w", encoding="utf-8"))
            )
        
        writer = csv.writer(f_out)
        writer.writerow(["id", "src", "tgt", "score", "status", "flag_reason"])

        for idx, (en_line, pt_line) in enumerate(zip(f_en, f_pt)):
            en_line_stripped = en_line.strip()
            pt_line_stripped = pt_line.strip()
            
            status = "clean"
            flag_reason = ""
            category = ""
            
            en_words = en_line_stripped.split()
            pt_words = pt_line_stripped.split()
            
            en_word_count = len(en_words)
            pt_word_count = len(pt_words)
            
            if en_line_stripped in seen_src:
                status = "flagged"
                flag_reason = "Duplicate source sentence"
                category = "duplicates"
            else:
                seen_src.add(en_line_stripped)
            
            if status == "clean":
                if en_word_count == 0 or pt_word_count == 0:
                    status = "flagged"
                    flag_reason = "Empty sentence"
                    category = "length_ratio"
                else:
                    ratio = pt_word_count / en_word_count
                    if ratio < 0.6 or ratio > 1.5:
                        status = "flagged"
                        flag_reason = f"Length ratio out of bounds ({ratio:.2f})"
                        category = "length_ratio"
            
            if status == "clean":
                if en_line_stripped == pt_line_stripped and en_word_count > 2:
                    status = "flagged"
                    flag_reason = "Identical strings (>2 words)"
                    category = "identical"

            if status == "clean":
                try:
                    en_lang, _ = model.predict(en_line_stripped.replace('\n', ' '))
                    pt_lang, _ = model.predict(pt_line_stripped.replace('\n', ' '))
                    
                    expected_src_label = f"__label__{config.SRC_LANG}"
                    expected_tgt_label = f"__label__{config.TGT_LANG}"
                    
                    if en_lang[0] != expected_src_label:
                        status = "flagged"
                        flag_reason = f"Source LangID failed (got {en_lang[0]})"
                        category = "LangID"
                    elif pt_lang[0] != expected_tgt_label:
                        status = "flagged"
                        flag_reason = f"Target LangID failed (got {pt_lang[0]})"
                        category = "LangID"
                except Exception as e:
                    status = "flagged"
                    flag_reason = f"LangID exception: {str(e)}"
                    category = "LangID"
            
            if status == "clean":
                if has_html_or_markdown(pt_line_stripped):
                    status = "flagged"
                    flag_reason = "HTML/Markdown detected in target"
                    category = "markdown"
                elif has_control_chars(pt_line_stripped):
                    status = "flagged"
                    flag_reason = "Control chars detected in target"
                    category = "control_char"
                elif '\ufffd' in pt_line_stripped:
                    status = "flagged"
                    flag_reason = "Mojibake/Replacement char detected"
                    category = "control_char"

            if status == "clean":
                f_clean_en.write(en_line)
                f_clean_pt.write(pt_line)
                clean_count += 1
            else:
                writer.writerow([idx, en_line_stripped, pt_line_stripped, "", status, flag_reason])
                f_cat_en, f_cat_pt = cat_files[category]
                f_cat_en.write(en_line)
                f_cat_pt.write(pt_line)
                flagged_count += 1
            
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1} lines...")

    print(f"Layer 1 complete. Clean sentences: {clean_count}, Flagged sentences: {flagged_count}.")
    print(f"Flagged results saved to {output_csv}.")
    
    print(f"Clean sentences saved to {clean_en_file} and {clean_pt_file}.")
    print("Done.")

if __name__ == "__main__":
    main()
