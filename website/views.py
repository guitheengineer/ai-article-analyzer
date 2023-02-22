from django.shortcuts import render
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
from collections.abc import Callable
from typing import Any

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained(
    "arpanghoshal/EmoRoBERTa")

# AI Model text limit
text_limit = 512

# Fix signal error when using extract.
config = use_config()
config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


def get_text_from_url(url: str) -> str | None:
    downloaded = fetch_url(url)
    article_text = extract(downloaded, config=config)
    return article_text


def get_aggregate_result(text: str, text_limit: int, classifier: Callable[[str], Any]):
    aggregate_result = list()

    for i in range(0, len(text) - text.count(' '), text_limit):
        text_chunk = text[i:i + text_limit]
        result = classifier(text_chunk)
        aggregate_result.append(result)

    return aggregate_result


def get_positiviness(text: str):
    classifier = pipeline(
        'sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    positive_results = get_aggregate_result(
        text, 512, classifier)
    result = sum(item[0]['label'] == 'POSITIVE' for item in positive_results)

    return result / len(positive_results) * 100


def get_summary(text: str):
    summarizer = pipeline(
        'summarization', model='philschmid/bart-large-cnn-samsum')
    summary_results = get_aggregate_result(text, 3500, summarizer)
    result = ' '.join([str(s[0]['summary_text']) for s in summary_results])

    return result


def get_emotion(text: str):

    emotion_classifier = pipeline(
        'sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    emotion = emotion_classifier(text)[0]['label']  # type: ignore
    return emotion


def index(request):
    search = request.GET.get('search')
    error = None
    context = {'search': search or '', 'error': error}
    if search:
        text = get_text_from_url(search)

        if not text:
            error = 'There is no content to analyze.'
            return

        summary = get_summary(text)
        context['summary'] = summary
        positiviness = get_positiviness(text)
        context['positiviness'] = positiviness
        # emotion = get_emotion(text)
        # context['emotion'] = emotion

    return render(request, 'website/index.html', context)
