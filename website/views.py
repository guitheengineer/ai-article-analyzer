from typing import Any
from django.shortcuts import render
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
from transformers import pipeline


def index(request):
    search = request.GET.get('search')
    context = {'search': search or ''}
    if search:
        try:
            hasHttp = search.startswith(
                'https://') or search.startswith('http://')
            url = search

            if not hasHttp:
                url = 'http://' + url

            page = urlopen(url)
            html = page.read().decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')
            articleText = ' '.join(soup.get_text().split())
            print(articleText)
            classifier = pipeline('sentiment-analysis')
            positivity = classifier(articleText[:512])
            context['positivity'] = positivity[0]['label']

        except requests.ConnectionError:
            print('Error: not able to connect to this website.')
    return render(request, 'website/index.html', context)
