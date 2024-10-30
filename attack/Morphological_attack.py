import random

def expand_contractions(text):
    # 常见缩略词的映射
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am"
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text

def random_upper_lower_flip(text, flip_rate=0.1):
    # 随机大写/小写字母翻转
    flipped_text = []
    for char in text:
        if char.isalpha() and random.random() < flip_rate:
            if char.islower():
                flipped_text.append(char.upper())
            else:
                flipped_text.append(char.lower())
        else:
            flipped_text.append(char)
    return ''.join(flipped_text)

# 示例文本
original_text = "I can't believe it's already here! This won't take long."
print("Original text:", original_text)

# 执行缩略词扩展
expanded_text = expand_contractions(original_text)
print("Text after expanding contractions:", expanded_text)

# 执行随机大小写翻转
obfuscated_text = random_upper_lower_flip(expanded_text, flip_rate=0.2)
print("Text after random upper/lower flip:", obfuscated_text)
