from django.shortcuts import render
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

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


def get_positiviness(text: str):
    positives = 0
    total = 0
    classifier = pipeline(
        'sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    for i in range(0, len(text), text_limit):
        text_chunk = text[i:i + text_limit]
        label = classifier(text_chunk)[0]['label']  # type: ignore
        total += 1
        if label == 'POSITIVE':
            positives += 1

    positiveness = positives / total * 100
    return positiveness


def get_summary(text: str):
    summarizer = pipeline(
        'summarization', model='philschmid/bart-large-cnn-samsum')
    result = summarizer(text)
    return result[0]['summary_text']  # type: ignore


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
        positiviness = get_positiviness(text)
        emotion = get_emotion(text)
        context['summary'] = summary
        context['positiviness'] = positiviness
        context['emotion'] = emotion

    return render(request, 'website/index.html', context)
