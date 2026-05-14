from pathlib import Path
import json

from tqdm import tqdm

def convert_to_markdown(sections):
    
    md = ""
    for sec in sections:
        if "header" in sec["type"]:
            md += f"<!--header-->\n## {sec['content']}\n\n"
        elif sec["type"] == "text":
            md += f"<!--text-->\n{sec['content']}\n\n"
        elif sec["type"] == "table":
            # Placeholder for table formatting
            md += f"<!--table-->\n{sec['content']}\n\n"
    return md
    

def clean_document(page):
    
    with open(page, "r") as f:
        parsed_doc = json.load(f)
    
    type_map = {
        "header": "header",
        "doc_title": "header",
        "paragraph_title": "header",
        "text": "text",
        "table": "table",
        "abstract": "text",
        "display_formula": "text",
        "display_formula": "text",
        "reference_content": "text",
        "content": "text",
        "algorithm": "text",
        "number": None,
        "aside_text": None,
        "header_image": None,
        "footer": None,
        "figure_title": None,
        "footnote": None,
        "image": None,
        "vision_footnote": None,
        "footer_image": None,
        "chart": None,
    }
    
    sections = [
        {
            "page_num": res["page_number"],
            "type": type_map[res["block_label"]],
            "content": res["block_content"],
        }
        for res in parsed_doc["parsing_res_list"]
        if type_map[res["block_label"]] is not None
    ]
    
    # correct for two-line headers
    for i, sec in enumerate(sections[1:], start=1):
        if sections[i-1]["type"] == "header" and sec["type"] == "header":
            sec["type"] = "merged_header"
            sec["content"] = sections[i-1]["content"] + " " + sec["content"]
            sections[i-1]["type"] = None      
    sections = [sec for sec in sections if sec["type"] is not None]
    
    # headers that start with a number are almost never actual headers
    for sec in sections:
        if sec["type"] == "header" and sec["content"][0].isdigit():
            sec["type"] = "text"
    
    return sections
    
def process_documents(input_dir, output_dir):
    
    documents = sorted(input_dir.glob("*/*.json"))
    n = len(documents)
    
    for doc in tqdm(documents, desc="Processing documents", total=n):
        doc_output_dir = output_dir / doc.parent.name
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        sections = clean_document(doc)
        md_content = convert_to_markdown(sections)
        md_path = doc_output_dir / (doc.stem + ".md")
        
        with open(doc_output_dir / (doc.stem + ".json"), "w") as f:
            json.dump(sections, f, indent=2)
        with open(md_path, "w") as f:
            f.write(md_content) 
    
def main():
    
    SOURCE = "cornell_retail_educ"
    
    input_dir = Path("cache/01_ocr_output") / SOURCE
    output_dir = Path("cache/02_segment_output") / SOURCE
    
    process_documents(input_dir, output_dir)
    
    
if __name__ == "__main__":
    main()