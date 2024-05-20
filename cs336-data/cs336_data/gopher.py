import nltk


def at_least_one_alpha(word: str) -> bool:
    for char in word:
        if char.isalpha():
            return True
    return False


def gopher_quality_filter(text: str) -> bool:
    words = nltk.word_tokenize(text)
    if len(words) < 50 or len(words) > 100_000:
        return False
    mean_word_length = sum(len(word) for word in words) / len(words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    lines = text.split("\n")
    ellipsis_lines = [line for line in lines if line.endswith("...")]
    if len(ellipsis_lines) / len(lines) > 0.3:
        return False
    alpha_words = [word for word in words if at_least_one_alpha(word)]
    if len(alpha_words) / len(words) < 0.8:
        return False
    return True
