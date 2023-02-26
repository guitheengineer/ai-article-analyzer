from django.shortcuts import render
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
from detoxify import Detoxify

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained(
    "arpanghoshal/EmoRoBERTa")

# AI Model text limit
text_limit = 2000

# Fix signal error when using extract.
config = use_config()
config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


def get_text_from_url(url: str) -> str | None:
    downloaded = fetch_url(url)
    article_text = extract(downloaded, config=config)
    return article_text


def get_positivity(text: str):
    classifier = pipeline(
        'sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    positive_results: str = classifier(text)[0]['label']  # type: ignore
    return positive_results.capitalize()


def get_summary(text: str):
    summarizer = pipeline(
        'summarization', model='facebook/bart-large-cnn')
    summary_results: str = summarizer(text)[0]['summary_text']  # type: ignore
    return summary_results.capitalize()


def get_emotion(text: str):
    emotion_classifier = pipeline(
        'sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    emotion: str = emotion_classifier(text)[0]['label']  # type: ignore
    return emotion.capitalize()


def get_toxicity(text: str):
    results = Detoxify('original').predict(text)
    return results


def get_chunked_text_infos(text: str):
    text_list = list()
    for i in range(0, len(text), text_limit):
        text_chunk = text[i:i + text_limit]
        text_dict = dict()
        text_dict['text'] = text_chunk
        text_dict['positivity'] = get_positivity(text_chunk)
        text_dict['summary'] = get_summary(text_chunk)
        text_dict['emotion'] = get_emotion(text_chunk)
        text_dict['toxicity'] = get_toxicity(text_chunk)
        text_list.append(text_dict)
    return text_list


def index(request):
    search = request.GET.get('search')
    error = None
    context = {'search': search or '', 'error': error}
    if search:
        text = get_text_from_url(search)

        if not text:
            error = 'There is no content to analyze.'
            return

        context['texts'] = get_chunked_text_infos(text)

    return render(request, 'website/index.html', context)
