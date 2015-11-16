import settings
import metadata
import nltk
import nltk.data
from gensim.models import word2vec
import math
import collections
import gensim
import io
import os
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

FILEPATH_RDF = os.environ.get("GUTENBERG_RDF_FILEPATH")
FILEPATH_GUTENBERG_TEXTS = os.environ.get("GUTENBERG_DATA")
FILEPATH_DATA = os.environ.get("DATA_SOURCE")
NLTK_DATA = os.environ.get("NLTK_DATA")
GUTENBERG_CORPUS = os.environ.get("NLTK_DATA_GUTENBERG_CORPUS")

w2v = None
def load_word2vec():
    global w2v
    w2v = word2vec.Word2Vec.load_word2vec_format(FILEPATH_DATA + os.sep + "word2vec" + os.sep + "GoogleNews-vectors-negative300.bin.gz",binary = True).similarity
#load_word2vec()

corpus_guten = nltk.corpus.reader.plaintext.PlaintextCorpusReader(GUTENBERG_CORPUS, fileids = r'.*\.txt')



















#class GutenbergCorpusView(StreamBackedCorpusView):
#    def __init__(self, *args, **kwargs):
#        StreamBackedCorpusView.__init__(self, *args, **kwargs)

##    def _open(self):
##        encoding = self._encoding
##        file_number = getNumberFromFilename(os.path.basename(self._fileid))
##        self._stream = io.StringIO(str("This is a test string. This is a test string. This is a test string."))
#            #gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(file_number)).strip())
#            #                       .encode(encoding="utf-8", errors="replace"))
            
#class GutenbergCorpusReader(PlaintextCorpusReader):
#    CorpusView = GutenbergCorpusView




##text = strip_headers(load_etext(2701)).strip()
##print(text)  # prints 'MOBY DICK; OR THE WHALE\n\nBy Herman Melville ...'


#sentences = [['first', 'sentence'], ['second', 'sentence'], ['this' 'is' 'also' 'a' 'sentence']]
##model = gensim.models.Word2Vec(sentences, min_count=1)


#w2v = None
##w2vbackup = None

#def load_word2vec():
#    global w2v
#    w2v = word2vec.Word2Vec.load_word2vec_format(FILEPATH_DATA + os.sep + "word2vec" + os.sep + "GoogleNews-vectors-negative300.bin.gz",binary = True).similarity
##    global w2vbackup
##    w2vbackup = word2vec.Word2Vec.load("data/gutenberg.w2v").similarity

#stopwords = None
#def load_stopwords():
#    global stopwords
#    if not stopwords:
#        stopwords = set(nltk.corpus.stopwords.words("english"))

#def text(filename):
#    return open(filename, mode="r", encoding="utf8")

#sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")

#def count_words(text):
#    load_stopwords()
#    collapsed = {}
#    total = 0
#    bycaps = collections.defaultdict(list)
#    source_text = text
#    words = nltk.wordpunct_tokenize(source_text)
#    words = [w for w in words if w.isalpha()]
#    count = collections.Counter(words)
#    for w,c in count.items():
#        bycaps[w.lower()].append((c, w))
#    for w, lst in bycaps.items():
#        canon = max(lst)[1]
#        if len(w) < 3 or w in stopwords:
#            continue
#        n = sum(x[0] for x in lst)
#        collapsed[w] = (n, canon)
#        total += n
#    return collapsed, total


##count_words(strip_headers(load_etext(123)).strip())

#def pos_tag_words(word, tags={}):
#    if not tags:
#        try:
#            print("Loading POS tags")
#            tags.update(json.load(open("data/pos.json")))
#        except IOError:
#            print("Counting POS tags")
#    try:
#        return tags[word.lower()]
#    except KeyError:
#        tag = nltk.pos_tag([word])[0][1]
#        if tag == "NNP": tag = "NP"
#        if tag == "NNPS": tag = "NPS"
#        tags[word.lower()] = tag
#        return tag

#ts1 = nltk.word_tokenize("This is a test sentence")
#ts2 = nltk.pos_tag(ts1)

#from gensim import corpora, models, similarities

#from nltk.corpus.reader import PlaintextCorpusReader
#from nltk.corpus.reader.util import StreamBackedCorpusView


#import io

#class GutenbergCorpusView(StreamBackedCorpusView):
#    def __init__(self, *args, **kwargs):
#        StreamBackedCorpusView.__init__(self, *args, **kwargs)

##    def _open(self):
##        encoding = self._encoding
##        file_number = getNumberFromFilename(os.path.basename(self._fileid))
##        self._stream = io.StringIO(str("This is a test string. This is a test string. This is a test string."))
#            #gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(file_number)).strip())
#            #                       .encode(encoding="utf-8", errors="replace"))
            
#class GutenbergCorpusReader(PlaintextCorpusReader):
#    CorpusView = GutenbergCorpusView
##    def open(self, file):
##        encoding = self.encoding(file)
##        file_number = getNumberFromFilename(os.path.basename(file))
#        #stream = io.StringIO(gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(file_number)).strip())
#        #stream = io.StringIO(u"This is a test string. This is a test string. This is a test string.")
#        #stream = io.BytesIO(gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(file_number)).strip().encode('ascii', 'replace'))
#        #stream = io.BytesIO("test")
#        #print(stream.encoding)
#        #stream.write(gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(file_number)).strip().encode(encoding, 'replace'))
                                  
##        return stream

##gutenberg_data_directory = "../../data/gutenberg_dvd_clean"
##corpus_gdvd = PlaintextCorpusReader(gutenberg_data_directory,)
##corpus_guten = corpusreaders.CompressedCorpusReader(FILEPATH_GUTENBERG_TEXTS + os.sep + "text", fileids = r'.*\.txt.gz')

#corpus_guten = GutenbergCorpusReader(FILEPATH_GUTENBERG_TEXTS + os.sep + "text", fileids = r'.*\.txt.gz', encoding='utf-8')



#def build_tags(*wcs):
#    tags = {}
#    for word_count in wcs:
#        for w in word_count.iterkeys():
#            tags[w] = pos_tag(word_count[w][1])
#    return tags

#def match(source_text, target_text, source_sentences, target_sentences):
#    source_wc = nltk.FreqDist(word.lower() for word in source_text)
#    target_wc = nltk.FreqDist(word.lower() for word in target_text)
#    translate = {}
#    source_fs = {}
#    target_fs = {}
#    print("Grouping POS")
#    for idx, wrd in enumerate(source_wc):
#        source_fs[wrd] = source_wc.freq(wrd)
#    for idx, wrd in enumerate(target_wc):
#        target_fs[wrd] = target_wc.freq(wrd)
#    #print(source_wc)
#    #print(source_fs)
#    source_by_freq = sorted(source_fs, key=lambda x: -source_fs[x])
#    target_by_freq = sorted(target_fs, key=lambda x: -target_fs[x])
#    #print(source_by_freq)
#    print("Matching Vocabulary")
#    translate = source_by_freq
#    source_sentences_tagged = nltk.pos_tag_sents(source_sentences)        
#    source_words_tagged = [item for sublist in source_sentences_tagged for item in sublist]
#    target_sentences_tagged = nltk.pos_tag_sents(target_sentences)        
#    target_words_tagged = [item for sublist in target_sentences_tagged for item in sublist]
#    translate = source_words_tagged
#    print("Translated")
#    return translate

##print(match(corpus_guten.words("10345.txt.gz"), corpus_guten.words("1345.txt.gz"), corpus_guten.sents("10345.txt.gz"), corpus_guten.sents("1345.txt.gz")))

