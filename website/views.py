from django.http import HttpResponse
from django.shortcuts import render
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
from django.template import Context


def index(request):
    search = request.GET.get('search')
    if search:
        try:
            # keywords = list()
            # keywordCount = dict()
            hasHttp = search.startswith(
                'https://') or search.startswith('http://')
            url = search

            if not hasHttp:
                url = 'http://' + url

            page = urlopen(url)
            html = page.read().decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')
            splittedWords = soup.get_text().split()
            print(splittedWords)
            for word in splittedWords:
                normalizedWord = word.lower()
                # if normalizedWord in keywords:
                # keywordCount[normalizedWord] = keywordCount.get(
                # normalizedWord, 0) + 1
                # print(keywordCount)
        except requests.ConnectionError:
            print('Error: not able to connect to this website.')
    return render(request, 'website/index.html', {'search': search})
