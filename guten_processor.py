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
        return int(filename.strip("pg.rdf"))
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


#from gutenberg.acquire import load_etext
#from gutenberg.cleanup import strip_headers
#text = strip_headers(load_etext(2701)).strip()
#print(text)  # prints 'MOBY DICK; OR THE WHALE\n\nBy Herman Melville ...'