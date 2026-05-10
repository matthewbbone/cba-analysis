
import json
import re
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

ROMAN_NUMERAL_PATTERN = re.compile(
    r"\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{1,3})\b",
    re.IGNORECASE,
)


def remove_roman_numerals(text):
    return ROMAN_NUMERAL_PATTERN.sub(" ", text)


def remove_standalone_letters(text):
    return re.sub(r"\b[a-z]\b", " ", text, flags=re.IGNORECASE)

def category_header_list(sections, category_dict):
    
    current_header = None
    for sec in sections:
        if sec["type"] in ["header", "merged_header"]:
            current_header = sec["content"]
        elif sec["type"] == "text" and current_header is not None:
            for p in sec["extracted_provisions"]:
                category = p.get("category")
                if category:
                    category_dict[category].append(current_header)
                    
    return category_dict
                    
def category_header_wordcloud(sources, input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open("references/provision_taxonomy.json", "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
        
    category_dict = {category["name"]: [] for category in taxonomy["categories"]}
    
    for source in sources:
        source_input_dir = input_dir / source
        for doc in source_input_dir.glob("*/*_classified.json"):
            with doc.open("r", encoding="utf-8") as f:
                sections = json.load(f)
            category_dict = category_header_list(sections, category_dict)
      
    for category, headers in category_dict.items():
        
        text = " ".join(headers).lower()
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.replace("section", " ")
        text = text.replace("article", " ")
        text = text.replace("appendix", " ")
        text = remove_roman_numerals(text)
        text = remove_standalone_letters(text)
        
        if len(text.strip()) == 0:
            continue
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            collocations=False,
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Category: {category}")
        
        output_path = output_dir / f"{category}_wordcloud.png"
        plt.savefig(output_path)
        plt.close()
    

def main():
    
    SOURCES = ["cornell_dol", "cornell_retail_educ", "dol_archive"]
    
    input_dir = Path("cache/04_classification_output")
    output_dir = Path("figures")
    
    category_header_wordcloud(SOURCES, input_dir, output_dir)
    
if __name__ == "__main__":
    main()
