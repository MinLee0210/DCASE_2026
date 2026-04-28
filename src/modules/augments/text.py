import random
import re

import nltk
from nltk.corpus import wordnet
from transformers import pipeline

# Download necessary nltk data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def get_synonyms(word):
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def synonym_replacement(text, n=1):
    """Replace n words in the text with their synonyms."""
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return " ".join(new_words)


def random_deletion(text, p=0.1):
    """Randomly delete words with probability p."""
    words = text.split()
    if len(words) == 1:
        return text
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return words[rand_int]
    return " ".join(new_words)


def random_swap(text, n=1):
    """Randomly swap two words in the text n times."""
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) < 2:
            break
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return " ".join(new_words)


def random_insertion(text, n=1):
    """Randomly insert n words into the text."""
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return " ".join(new_words)


def add_word(new_words):
    """Helper function to add a random word."""
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def back_translation(text, src_lang="en", intermediate_lang="fr"):
    """Translate to intermediate language and back."""
    try:
        translator = pipeline(
            "translation", model=f"Helsinki-NLP/opus-mt-{src_lang}-{intermediate_lang}"
        )
        back_translator = pipeline(
            "translation", model=f"Helsinki-NLP/opus-mt-{intermediate_lang}-{src_lang}"
        )

        translated = translator(text)[0]["translation_text"]
        back_translated = back_translator(translated)[0]["translation_text"]
        return back_translated
    except Exception as e:
        print(f"Back translation failed: {e}")
        return text


def contextual_augmentation(text, model_name="distilgpt2"):
    """Use a language model to generate augmented text."""
    try:
        generator = pipeline("text-generation", model=model_name)
        # Simple approach: generate continuation and take part of it
        generated = generator(
            text, max_length=len(text.split()) + 10, num_return_sequences=1
        )[0]["generated_text"]
        # Extract a similar length text
        words = generated.split()
        original_len = len(text.split())
        return " ".join(words[:original_len])
    except Exception as e:
        print(f"Contextual augmentation failed: {e}")
        return text


def augment_text(text, techniques=None, **kwargs):
    """Apply multiple augmentation techniques."""
    if techniques is None:
        techniques = [
            "synonym_replacement",
            "random_deletion",
            "random_swap",
            "random_insertion",
        ]

    augmented = text
    for tech in techniques:
        if tech == "synonym_replacement":
            n = kwargs.get("synonym_n", 1)
            augmented = synonym_replacement(augmented, n)
        elif tech == "random_deletion":
            p = kwargs.get("deletion_p", 0.1)
            augmented = random_deletion(augmented, p)
        elif tech == "random_swap":
            n = kwargs.get("swap_n", 1)
            augmented = random_swap(augmented, n)
        elif tech == "random_insertion":
            n = kwargs.get("insertion_n", 1)
            augmented = random_insertion(augmented, n)
        elif tech == "back_translation":
            src = kwargs.get("src_lang", "en")
            inter = kwargs.get("intermediate_lang", "fr")
            augmented = back_translation(augmented, src, inter)
        elif tech == "contextual_augmentation":
            model = kwargs.get("model_name", "distilgpt2")
            augmented = contextual_augmentation(augmented, model)

    return augmented
