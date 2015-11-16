import settings
import rdflib
import gutenberg
import os
import xml.etree.cElementTree as ElementTree
from rdflib import plugin, Graph, Literal, URIRef
from rdflib.store import Store


# Grab environment variables...
FILEPATH_RDF = os.environ.get("GUTENBERG_RDF_FILEPATH")
FILEPATH_GUTENBERG_TEXTS = os.environ.get("GUTENBERG_DATA")
FILEPATH_DATA = os.environ.get("DATA_SOURCE")

def openRDF(rdf_name) -> rdflib.Graph:
    """
    Finds and opens a PG RDF file from the data directory.
    Can take either a number, in which case it will translate
    it to a file name, or a string with the exact file name, 
    which it will use verbatum. 
    """
    g = rdflib.Graph()
    if not isinstance(rdf_name, str):
        rdf_name = "pg" + str(rdf_name) + ".rdf"
    filepath = FILEPATH_RDF + os.sep + str(rdf_name)
    if not os.path.isfile(filepath):
        #print("Warning: " + str(filepath) + " does not exist")
        return
    g.open(FILEPATH_GUTENBERG_TEXTS + os.sep + "rdflib_db" + os.sep + "db", create = True)
    g.load(filepath)
    g.close()
    return g

META_FIELDS = ('id', 'author', 'title', 'downloads', 'formats', 'type', 'LCC',
		'subjects', 'authoryearofbirth', 'authoryearofdeath', 'language', 'storedlocally')
NS = dict(
		pg='http://www.gutenberg.org/2009/pgterms/',
		dc='http://purl.org/dc/terms/',
		dcam='http://purl.org/dc/dcam/',
		rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#')

def getNumberFromFilename(filename):
    if isinstance(filename, str):
        return int(filename.strip("pg.rdftxtgz"))
    if isinstance(filename, int):
        return filename
    raise TypeError

def getFilenameFromNumber(filenumber):
    if isinstance(filenumber, str):
        return filenumber # todo: check if valid filename
    if isinstance(filenumber, int):
        return "pg" + str(filenumber) + ".rdf"
    raise TypeError

#filepath = FILEPATH_RDF + "/" + str(getFilenameFromNumber(text_id))
#with open(filepath, 'r') as file:
#    element_tree = ElementTree.parse(file)
#if element_tree:
#    print(element_tree.find("creator"))

def getDataNodeFromMetadata(rdf, data_type):
    data = [rdf.triples((i[2], rdflib.term.URIRef(NS['rdf'] + "value"), None)) for i in rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + data_type), None))]
    values = [str(j[2].toPython()) for i in data for j in i]
    result = values
    return result

def getMetadataForText(text_id):
    """
    Given a text ID number or rdf filename, return the metadata.
    """
    #print(str(text_id))
    rdf = openRDF(text_id)
    if not rdf:
        return
    #print(str("opening..."))
    #element_tree = None
    #ebook_metadata = ElementTree.parse(rdf)
    result = dict.fromkeys(META_FIELDS)
    result['id'] = getNumberFromFilename(text_id)
    # authors' names
    creator_data = rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "creator"), None))
    authors_data = [rdf.triples((i[2], rdflib.term.URIRef(NS['pg'] + "name"), None)) for i in creator_data]
    authors_names = [j[2].toPython() for i in authors_data for j in i]
    result['author'] = authors_names
    # authors' numbers
    result['agents'] = [os.path.basename(x[2].toPython()) for x in rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "creator"), None))]
    # title
    result['title'] = [i[2].toPython() for i in rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "title"), None))]
    if 1 == len(result['title']):
        result['title'] = result['title'][0]
    # LCC
    memberOf_data = [rdf.triples((i[2], rdflib.term.URIRef(NS['dcam'] + "memberOf"), None)) for i in rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "subject"), None))]
    bnodes = [j[0] for i in memberOf_data for j in i if j[2] == rdflib.term.URIRef(NS['dc'] + "LCC")]
    value_data = [str(j[2].toPython()) for i in [rdf.triples( (i, rdflib.term.URIRef(NS['rdf'] + "value"), None)) for i in bnodes] for j in i]
    result['LCC'] = value_data
    # Subjects
    memberOf_data = [rdf.triples((i[2], rdflib.term.URIRef(NS['dcam'] + "memberOf"), None)) for i in rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "subject"), None))]
    bnodes = [j[0] for i in memberOf_data for j in i if j[2] == rdflib.term.URIRef(NS['dc'] + "LCSH")]
    value_data = [str(j[2].toPython()) for i in [rdf.triples( (i, rdflib.term.URIRef(NS['rdf'] + "value"), None)) for i in bnodes] for j in i]
    result['subjects'] = value_data
    # languages
    result['language'] = getDataNodeFromMetadata(rdf, "language")
    # type
    result['type'] = getDataNodeFromMetadata(rdf, "type")[0]
    # formats
    #data = [rdf.triples((i[2], rdflib.term.URIRef(NS['rdf'] + "value"), None)) for i in rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "format"), None))]
    #values = [str(j[2].toPython()) for i in data for j in i]
    result['formats'] = getDataNodeFromMetadata(rdf, "format")
    # rights
    data = [rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "rights"), None))]
    values = [str(j[2].toPython()) for i in data for j in i]
    result['rights'] = values
    # do we have it on disk?

    if "Text" == result['type']:
        filename = FILEPATH_GUTENBERG_TEXTS + os.sep + "text" + os.sep + str(text_id) + ".txt.gz"
        result['filename'] = filename

        if os.path.isfile(filename):
            if os.path.exists(filename):
                result['storedlocally'] = True
        filename = FILEPATH_GUTENBERG_TEXTS + os.sep + "text" + os.sep + str(text_id) + ".txt.gz"
        if os.path.isfile(filename) and os.path.exists(filename):
            result['storedlocally'] = True
        else:
            #if 'text/plain; charset=us-ascii' in result['formats']:
            try_filename = FILEPATH_GUTENBERG_TEXTS + os.sep + "text" + os.sep + str(text_id) + "-8.txt.gz"
            if os.path.isfile(try_filename) and os.path.exists(try_filename):
                result['storedlocally'] = True
                filename = try_filename
            #if "text/plain; charset=utf-8" in result['formats']:
            try_filename = FILEPATH_GUTENBERG_TEXTS + os.sep + "text" + os.sep + str(text_id) + "-0.txt.gz"
            if os.path.isfile(try_filename) and os.path.exists(try_filename):
                result['storedlocally'] = True
                filename = try_filename
        result['filename'] = filename # we only care about texts...
    
    #print(str(text_id) + " (" + str(result["type"])+ "): " + str(result['storedlocally']) + " - " + str(result['title']).replace("\n",":"))
    return result

#print(getMetadataForText(18))
def testingLogFiles():
    with open("current_local_text_files.txt", "w" , encoding="utf-8") as output_file:
        [output_file.write(
            str(
                str(x['id']) 
                #+ " (" + str(x["type"]) 
                #+ "): " + str(x['rights']) 
                #+ " " + str(x['storedlocally']) 
                #+ " - " + str(x['title']).replace("\n",":").replace("\r",":") 
                + ","
                ))
        for x in [getMetadataForText(i) for i in range(0, 51000)] 
        if (x != None) and ("Text" == x['type']) and ('en' in x['language']) and ('Public domain in the USA.' in x['rights']) and (x['storedlocally'] == True)]

def testingMetadataFiles():
    [x['id']
        for x in [getMetadataForText(i) for i in settings.LIVE_FILE_LIST] 
        if (x != None) and ("Text" == x['type']) and ('en' in x['language']) and ('Public domain in the USA.' in x['rights']) and (x['storedlocally'] == True)]

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
#text = strip_headers(load_etext(2701)).strip()
#print(text)  # prints 'MOBY DICK; OR THE WHALE\n\nBy Herman Melville ...'

import gensim
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence'], ['this' 'is' 'also' 'a' 'sentence']]
#model = gensim.models.Word2Vec(sentences, min_count=1)

import nltk
import nltk.data
import nltk.tokenize
from gensim.models import word2vec
import math
import collections
w2v = None
#w2vbackup = None

def load_word2vec():
    global w2v
    w2v = word2vec.Word2Vec.load_word2vec_format(FILEPATH_DATA + os.sep + "word2vec" + os.sep + "GoogleNews-vectors-negative300.bin.gz",binary = True).similarity
#    global w2vbackup
#    w2vbackup = word2vec.Word2Vec.load("data/gutenberg.w2v").similarity

stopwords = None
def load_stopwords():
    global stopwords
    if not stopwords:
        stopwords = set(nltk.corpus.stopwords.words("english"))

def text(filename):
    return open(filename, mode="r", encoding="utf8")

sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")

def count_words(text):
    load_stopwords()
    collapsed = {}
    total = 0
    bycaps = collections.defaultdict(list)
    source_text = text
    words = nltk.wordpunct_tokenize(source_text)
    words = [w for w in words if w.isalpha()]
    count = collections.Counter(words)
    for w,c in count.items():
        bycaps[w.lower()].append((c, w))
    for w, lst in bycaps.items():
        canon = max(lst)[1]
        if len(w) < 3 or w in stopwords:
            continue
        n = sum(x[0] for x in lst)
        collapsed[w] = (n, canon)
        total += n
    return collapsed, total


#count_words(strip_headers(load_etext(123)).strip())

def pos_tag_words(word, tags={}):
    if not tags:
        try:
            print("Loading POS tags")
            tags.update(json.load(open("data/pos.json")))
        except IOError:
            print("Counting POS tags")
    try:
        return tags[word.lower()]
    except KeyError:
        tag = nltk.pos_tag([word])[0][1]
        if tag == "NNP": tag = "NP"
        if tag == "NNPS": tag = "NPS"
        tags[word.lower()] = tag
        return tag

ts1 = nltk.word_tokenize("This is a test sentence")
ts2 = nltk.pos_tag(ts1)

from gensim import corpora, models, similarities

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView


import io

class GutenbergCorpusView(StreamBackedCorpusView):
    def __init__(self, *args, **kwargs):
        StreamBackedCorpusView.__init__(self, *args, **kwargs)

#    def _open(self):
#        encoding = self._encoding
#        file_number = getNumberFromFilename(os.path.basename(self._fileid))
#        self._stream = io.StringIO(str("This is a test string. This is a test string. This is a test string."))
            #gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(file_number)).strip())
            #                       .encode(encoding="utf-8", errors="replace"))
            
class GutenbergCorpusReader(PlaintextCorpusReader):
    CorpusView = GutenbergCorpusView
#    def open(self, file):
#        encoding = self.encoding(file)
#        file_number = getNumberFromFilename(os.path.basename(file))
        #stream = io.StringIO(gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(file_number)).strip())
        #stream = io.StringIO(u"This is a test string. This is a test string. This is a test string.")
        #stream = io.BytesIO(gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(file_number)).strip().encode('ascii', 'replace'))
        #stream = io.BytesIO("test")
        #print(stream.encoding)
        #stream.write(gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(file_number)).strip().encode(encoding, 'replace'))
                                  
#        return stream

#gutenberg_data_directory = "../../data/gutenberg_dvd_clean"
#corpus_gdvd = PlaintextCorpusReader(gutenberg_data_directory,)
#corpus_guten = corpusreaders.CompressedCorpusReader(FILEPATH_GUTENBERG_TEXTS + os.sep + "text", fileids = r'.*\.txt.gz')

corpus_guten = GutenbergCorpusReader(FILEPATH_GUTENBERG_TEXTS + os.sep + "text", fileids = r'.*\.txt.gz', encoding='utf-8')



def build_tags(*wcs):
    tags = {}
    for word_count in wcs:
        for w in word_count.iterkeys():
            tags[w] = pos_tag(word_count[w][1])
    return tags

def match(source_text, target_text, source_sentences, target_sentences):
    source_wc = nltk.FreqDist(word.lower() for word in source_text)
    target_wc = nltk.FreqDist(word.lower() for word in target_text)
    translate = {}
    source_fs = {}
    target_fs = {}
    print("Grouping POS")
    for idx, wrd in enumerate(source_wc):
        source_fs[wrd] = source_wc.freq(wrd)
    for idx, wrd in enumerate(target_wc):
        target_fs[wrd] = target_wc.freq(wrd)
    #print(source_wc)
    #print(source_fs)
    source_by_freq = sorted(source_fs, key=lambda x: -source_fs[x])
    target_by_freq = sorted(target_fs, key=lambda x: -target_fs[x])
    #print(source_by_freq)
    print("Matching Vocabulary")
    translate = source_by_freq
    source_sentences_tagged = nltk.pos_tag_sents(source_sentences)        
    source_words_tagged = [item for sublist in source_sentences_tagged for item in sublist]
    target_sentences_tagged = nltk.pos_tag_sents(target_sentences)        
    target_words_tagged = [item for sublist in target_sentences_tagged for item in sublist]
    translate = source_words_tagged
    print("Translated")
    return translate

#print(match(corpus_guten.words("10345.txt.gz"), corpus_guten.words("1345.txt.gz"), corpus_guten.sents("10345.txt.gz"), corpus_guten.sents("1345.txt.gz")))

[x for x in corpus_guten.words("123.txt")]
