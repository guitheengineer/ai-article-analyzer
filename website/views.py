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
        classifier = pipeline('sentiment-analysis')

        for i in range(0, len(text), text_limit):
            text_chunk = text[i:i + text_limit]
            result = classifier(text_chunk)
            result_label = result[0]['label']
            total += 1
            if result_label == 'POSITIVE':
                positives += 1

        positiveness = positives / total * 100

        context['positiviness'] = positiveness

    return render(request, 'website/index.html', context)
