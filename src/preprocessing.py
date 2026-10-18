import emoji
import re

def preprocess_text(text: str) -> str:
    """
    Converts emojis into descriptive text tokens (e.g., 💀 -> emoji_skull)
    and cleans extra whitespace and casing.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Demojize emojis to descriptive text tokens
    text_demojized = emoji.demojize(text, delimiters=(" emoji_", " "))
    
    # 2. Lowercase and clean extra whitespace
    clean_text = text_demojized.lower().strip()
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text

if __name__ == "__main__":
    # Test cases for context-aware Gen Z slang & emojis
    test_samples = [
        "I'm dead 💀",
        "you ate",
        "that's sick 🔥",
        "I'm crying 😭"
    ]
    
    print("\n--- Context-Aware Preprocessing Test ---")
    for sample in test_samples:
        print(f"Original : {sample}")
        print(f"Processed: {preprocess_text(sample)}\n")