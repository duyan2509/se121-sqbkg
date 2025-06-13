import json
from underthesea import sent_tokenize, chunk
import requests

def load_stopwords():
    url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
    response = requests.get(url)
    stopwords = set()
    if response.status_code == 200:
        for line in response.text.splitlines():
            stopwords.add(line.strip())
    return stopwords

def remove_stopwords(words, stopwords):
    return [word for word in words if word.lower() not in stopwords]


def extract_chunks(text, stopwords):
    text = text.lower()
    text = text.replace("phòng, chống", 'phòng chống')
    noun_phrases = []
    verb_phrases = []
    sentences = sent_tokenize(text)

    for sentence in sentences:
        chunked = chunk(sentence)

        current_phrase = []
        current_type = None
        prev_tag = None  # Lưu trữ tag trước đó

        for word, pos, chunk_tag in chunked:
            if chunk_tag == "B-NP" and current_type == "NP" and prev_tag in ["B-NP", "I-NP"]:
                current_phrase.append(word)
                prev_tag = chunk_tag
                continue

            if chunk_tag.startswith("B-"):
                if current_phrase and current_type:
                    phrase = " ".join(current_phrase)
                    if current_type == "NP":
                        noun_phrases.append(phrase)
                    elif current_type == "VP":
                        verb_phrases.append(phrase)

                current_phrase = [word]
                current_type = chunk_tag[2:]

            elif chunk_tag.startswith("I-") and current_type == chunk_tag[2:]:
                current_phrase.append(word)

            else:
                if current_phrase and current_type:
                    phrase = " ".join(current_phrase)
                    if current_type == "NP":
                        noun_phrases.append(phrase)
                    elif current_type == "VP":
                        verb_phrases.append(phrase)

                current_phrase = []
                current_type = None

            prev_tag = chunk_tag

        if current_phrase and current_type:
            phrase = " ".join(current_phrase)
            if current_type == "NP":
                noun_phrases.append(phrase)
            elif current_type == "VP":
                verb_phrases.append(phrase)

    noun_phrases = remove_stopwords(noun_phrases, stopwords)
    verb_phrases = remove_stopwords(verb_phrases, stopwords)

    noun_phrases = [phrase.lower() for phrase in noun_phrases]
    verb_phrases = [verb_phrase.lower() for verb_phrase in verb_phrases]

    return noun_phrases, verb_phrases

def extract_chunks1(text, stopwords):
    text = text.lower().replace("phòng, chống", 'phòng chống')
    noun_phrases = []
    verb_phrases = []
    sentences = sent_tokenize(text)

    for sentence in sentences:
        chunked = chunk(sentence)

        current_phrase = []
        current_type = None

        for word, pos, chunk_tag in chunked:
            if chunk_tag.startswith("B-"):
                if current_phrase and current_type:
                    phrase = " ".join(current_phrase)
                    if current_type == "NP":
                        noun_phrases.append(phrase)
                    elif current_type == "VP":
                        verb_phrases.append(phrase)

                current_phrase = [word]
                current_type = chunk_tag[2:]

            elif chunk_tag.startswith("I-") and current_type == chunk_tag[2:]:
                current_phrase.append(word)

            else:
                if current_phrase and current_type:
                    phrase = " ".join(current_phrase)
                    if current_type == "NP":
                        noun_phrases.append(phrase)
                    elif current_type == "VP":
                        verb_phrases.append(phrase)
                current_phrase = []
                current_type = None

        if current_phrase and current_type:
            phrase = " ".join(current_phrase)
            if current_type == "NP":
                noun_phrases.append(phrase)
            elif current_type == "VP":
                verb_phrases.append(phrase)

    noun_phrases = remove_stopwords(noun_phrases, stopwords)
    verb_phrases = remove_stopwords(verb_phrases, stopwords)

    noun_phrases = [phrase.strip().lower() for phrase in noun_phrases]
    verb_phrases = [phrase.strip().lower() for phrase in verb_phrases]

    return noun_phrases, verb_phrases


def process_sentences(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # stopwords = load_stopwords()
    # stopwords.remove("người")
    stopwords=[]
    print(stopwords)
    results = []
    for entry in data:
        sentence = entry["sentence"]
        noun_phrases, verb_phrases = extract_chunks(sentence,stopwords)

        result = {
            "sentence": sentence,
            "concepts": noun_phrases,
            "relations": verb_phrases,
            "reference": entry.get("reference", {})
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_sentences("sentences.json", "sentence_with_cr_underthesea.json")
