from django.shortcuts import render
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
from transformers import pipeline

# AI Model text limit
text_limit = 512

# Fix signal error when using extract.
config = use_config()
config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


def get_text_from_url(url: str) -> str | None:
    downloaded = fetch_url(url)
    article_text = extract(downloaded, config=config)
    return article_text


def measure_positivity(text: str):
    classifier = pipeline(
        'sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    label = classifier(text)[0]['label']  # type: ignore
    return label


def summarize(text: str):
    summarizer = pipeline(
        'summarization', model='philschmid/bart-large-cnn-samsum')
    result = summarizer(text)
    return result[0]['summary_text']  # type: ignore


def index(request):
    search = request.GET.get('search')
    error = None
    context = {'search': search or '', 'error': error}
    if search:
        text = get_text_from_url(search)

        if not text:
            error = 'There is no content to analyze.'
            return
        positives = 0
        total = 0

        for i in range(0, len(text), text_limit):
            text_chunk = text[i:i + text_limit]
            result = measure_positivity(text_chunk)
            total += 1
            if result == 'POSITIVE':
                positives += 1

        positiveness = positives / total * 100

        summary = summarize(text)
        context['positiviness'] = positiveness
        context['summary'] = summary

    return render(request, 'website/index.html', context)
