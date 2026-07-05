
import json
import re
from pathlib import Path
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.utils.llm import model_slug

ROMAN_NUMERAL_PATTERN = re.compile(
    r"\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{1,3})\b",
    re.IGNORECASE,
)


def remove_roman_numerals(text):
    return ROMAN_NUMERAL_PATTERN.sub(" ", text)


def remove_standalone_letters(text):
    return re.sub(r"\b[a-z]\b", " ", text, flags=re.IGNORECASE)


def looks_like_section_title(section):
    content = section.get("content", "").strip()
    if not content or len(content) > 160:
        return False
    if section.get("category") or section.get("is_provision"):
        return False

    normalized = re.sub(r"\s+", " ", content)
    if re.match(r"^(article|appendix|section)\b", normalized, flags=re.IGNORECASE):
        return True

    letters = [char for char in normalized if char.isalpha()]
    if not letters:
        return False

    uppercase_share = sum(char.isupper() for char in letters) / len(letters)
    return uppercase_share >= 0.7 and "\n" not in content


def category_section_title_list(sections, category_dict):
    current_title = None
    for sec in sections:
        section_type = sec.get("type")
        if section_type in ["header", "merged_header"] or looks_like_section_title(sec):
            current_title = sec.get("content", "").strip()
        elif section_type == "text" and current_title is not None:
            category = sec.get("category")
            if category:
                category_dict[category].append(current_title)
                    
    return category_dict
                    
def category_section_title_wordcloud(sources, input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open("references/provision_taxonomy.json", "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
        
    category_dict = {category["name"]: [] for category in taxonomy["categories"]}
    
    for source in sources:
        source_input_dir = input_dir / source
        for doc in source_input_dir.glob("*.json"):
            with doc.open("r", encoding="utf-8") as f:
                sections = json.load(f)
            category_dict = category_section_title_list(sections, category_dict)
      
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
            width=1200,
            height=600,
            background_color="white",
            collocations=False,
            max_font_size=180,
            min_font_size=18,
            relative_scaling=0.65,
        ).generate(text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(
            category,
            fontsize=26,
            fontweight="bold",
            pad=18,
        )
        
        output_path = output_dir / f"{category}_wordcloud.png"
        plt.savefig(output_path)
        plt.close()
    

def main():
    
    SOURCES = ["cornell_dol", "cornell_retail_educ", "dol_archive"]
    MODEL_NAME = "gpt-5.4-nano"
    
    input_dir = Path("cache/04_classification_output") / model_slug(MODEL_NAME)
    output_dir = Path("figures")
    
    category_section_title_wordcloud(SOURCES, input_dir, output_dir)
    
if __name__ == "__main__":
    main()
