import string

# Basic stopword list
stop_words = {
    "the", "and", "is", "in", "it", "of", "to", "a", "this", "that", "was", "for",
    "on", "with", "as", "but", "are", "not", "be", "at", "by", "an", "or", "from"
}

def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation and not c.isdigit())
    return [w for w in text.split() if w not in stop_words]

def clean_text_joined(text):
    return " ".join(clean_text(text))
