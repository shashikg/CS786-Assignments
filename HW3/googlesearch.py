import urllib.request as urllib2
from bs4 import BeautifulSoup
import numpy as np

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36"
HEADERS = {'User-Agent' : USER_AGENT}

def get_freq(search_query):
    search_url =  'http://www.google.com/search?q=' + search_query + '&start=1'
    
    request = urllib2.Request(search_url, None, HEADERS)
    response = urllib2.urlopen(request)
    html = response.read()
    
    soup = BeautifulSoup(html, "lxml")
    f_text = soup.find('div',{'id':'resultStats'}).text #search for the resultStatus ID which contains the page found
    
    # "Page 2 of about 2,41,00,000 result" is the string pattern inside the resultStaus
    # So 17 index of the string starts the count of no of pages found.
    # So count was don till next space
    i = 17
    n = f_text[16]
    while 1:
        if f_text[i] == ' ':
            break

        if f_text[i] != ',':
            n += f_text[i]
        i += 1

    f = eval(n)
    
    return f
    