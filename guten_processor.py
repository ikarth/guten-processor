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
import json
import re
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

stopwords = None
def load_stopwords():
    global stopwords
    if not stopwords:
        stopwords = set(nltk.corpus.stopwords.words("english"))

sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")

def get_text(text_number):
    """
    Given a text number, return the filename.
    """
    meta = metadata.getMetadataForText(text_number)
    if not meta['filename']:
        return
    return os.path.basename(meta['filename'])

def count_words(text):
    """
    Given a text string, count the number of unique words.
    Returns a dict with the kay as the (lowercased) words,
    and the value as a tuple of the count and the word.
    Also returns the total number of unique words.
    Doesn't count words in the stoplist.

    (text -> dict, word-count)
    """
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
        if len(w) < 2 or w in stopwords:
            continue
        n = sum(x[0] for x in lst)
        collapsed[w] = (n, canon)
        total += n
    return collapsed, total

##count_words(strip_headers(load_etext(123)).strip())

def pos_tag_words(word, tags={}):
    """
    Returns a POS tag for a word passed to it,
    usually from a word count list.
    """
    if not tags:
        try:
            print("Loading POS tags")
            tags.update(json.load(open("data/pos.json")))
            print("Loaded")
        except IOError:
            print("Counting POS tags")
            tagcounts = collections.defaultdict(collections.Counter)
            bonuses = collections.defaultdict(int)
            bonuses["VBG"] = 1
            for w, t in nltk.corpus.brown.tagged_words():
                if not w.isalpha(): continue
                t = t.split('-')[0]
                tagcounts[w.lower()][t] += 1 + bonuses[t]
            for w in iter(tagcounts.keys()):
                tags[w] = tagcounts[w].most_common(1)[0][0]
            json.dump(tags, open("data/pos.json", "w"))
            print("Counted")
    try:
        return tags[word.lower()]
    except KeyError:
        print(str(word))
        tag = nltk.pos_tag([word])[0][1]
        if tag == "NNP": tag = "NP"
        if tag == "NNPS": tag = "NPS"
        tags[word.lower()] = tag
        return tag

def semantic_sim(word_one, word_two):
    """
    Takes two words, returns their semantic similarity 
    based on their word2vec results.
    """
    if not w2v:
        load_word2vec()
    try:
        return w2v(word_one, word_two)
    except KeyError:
        pass
    return 0.0

def build_tags(*wcs):
    """
    Build a list of tags for the words in the passed-in
    word count string(s).
    """
    tags = {}
    idx = 0
    for word_count in wcs:
        idx += 1
        if idx % 100 == 0:
            print(idx)
        for w in iter(word_count.keys()):
            tags[w] = pos_tag_words(word_count[w][1])
    return tags

def match(source, target):
    source_text = corpus_guten.raw(source)
    target_text = corpus_guten.raw(target)
    source_sentences = corpus_guten.sents(source)
    target_sentences = corpus_guten.sents(target)

    source_wc, source_len = count_words(source_text)# nltk.FreqDist(word.lower() for word in source_text)
    target_wc, target_len = count_words(target_text)# nltk.FreqDist(word.lower() for word in target_text)
    translate = {}
    source_freqs = {}
    target_freqs = {}
    print("Grouping POS")
    tags = build_tags(source_wc, target_wc)
    for idx, wrd in enumerate(source_wc):
        if idx % 10 == 0:
            print(idx)
        source_freqs[wrd] = math.log(source_wc[wrd][0]/source_len)
    for idx, wrd in enumerate(target_wc):
        if idx % 10 == 0:
            print(idx)
        target_freqs[wrd] = math.log(target_wc[wrd][0]/target_len)
    source_by_freq = sorted(source_freqs.keys(), key=lambda x: -source_freqs[x])
    target_by_freq = sorted(target_freqs.keys(), key=lambda x: -target_freqs[x])
    print("Matching vocabulary")
    maxsem = 2
    minsem = -2
    good_tags = ["NN", "NP", "NNS", "NPS", "VB", "VBD", "VBZ", "VBG", "VBN", "JJ"]
    penalties = collections.defaultdict(float)
    for i, source_word in enumerate(source_by_freq):
        if i % 1000 == 0:
            print(i)
        if tags[source_word] not in good_tags: continue
        if source_word in target_freqs:
            bestscore = minsem + (source_freqs[source_word] - target_freqs[source_word]) ** 2 + penalties[source_word]
        else:
            bestscore = 1e9
        best = source_word
        for target_word in target_by_freq:
            if tags[source_word] != tags[target_word]: continue
            freqscore = (source_freqs[source_word] - target_freqs[target_word]) ** 2
            if freqscore + minsem > bestscore:
                if target_freqs[target_word] < source_freqs[source_word]: break
                continue
            semanticscore = -2 * semantic_sim(source_wc[source_word][1], target_wc[target_word][1])
            score = freqscore + semanticscore + penalties[target_word]
            if score < bestscore:
                best = target_word
                bestscore = score
        penalties[best] += math.log(source_wc[source_word][0])
        if best != source_word:
            translate[source_word] = best
        if i < 100:
            print(source_word, best)
    return translate

def fix_articles(txt):
    def fixup(match):
        oldart = match.group(0)
        spacing = match.group(2)
        firstchar = match.group(3)
        if firstchar in 'aeiouAEIOU':
            article = 'an'
        else:
            article = 'a'
        if oldart[0].isupper():
            article = article.capitalize()
        return article + spacing + firstchar
    return re.sub(r'\b(a|an)(\s+)([a-z])', fixup, txt, flags=re.IGNORECASE)

def translate(filename, translation):
    txt = corpus_guten.raw(filename)
    def replace_word(match):
        word = match.group(0)
        try:
            repword = translation[word.lower()]
        except KeyError:
            return word
        if word.isupper():
            repword = repword.upper()
        elif word[0].isupper():
            repword = repword.capitalize()
        else:
            repword = repword.lower()
        return repword
    regex = re.compile(r'\w+|[^\w\s]+')
    print ("Translating")
    newtxt = regex.sub(replace_word, txt)
    return fix_articles(newtxt)

def translate_match(structure, vocab):
    return translate(structure, match(structure, vocab))

#    print("Matching Vocabulary")
#    translate = source_by_freq
#    source_sentences_tagged = nltk.pos_tag_sents(source_sentences)        
#    source_words_tagged = [item for sublist in source_sentences_tagged for item in sublist]
#    target_sentences_tagged = nltk.pos_tag_sents(target_sentences)        
#    target_words_tagged = [item for sublist in target_sentences_tagged for item in sublist]
#    translate = source_words_tagged
#    print("Translated")
#    return translate



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





##print(match(corpus_guten.words("10345.txt.gz"), corpus_guten.words("1345.txt.gz"), corpus_guten.sents("10345.txt.gz"), corpus_guten.sents("1345.txt.gz")))

#with open('alice_four.txt', 'w') as file:
    #for i in corpus_guten.words('11.txt'):#(nltk.word_tokenize(corpus_guten.raw('11.txt'))):
     #   file.write("%s\n" % i)
   #file.write(corpus_guten.words('11.txt'))
   #print(corpus_guten._word_tokenizer())