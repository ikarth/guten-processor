import settings
import rdflib
import gutenberg
import os
import xml.etree.cElementTree as ElementTree

# Grab environment variables...
FILEPATH_RDF = os.environ.get("GUTENBERG_RDF_FILEPATH")

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
    filepath = FILEPATH_RDF + "/" + str(rdf_name)
    g.open("db")
    g.load(filepath)
    #for s, p, o in g:
    #    print(s)
    #    print(p)
    #    print(o)
    #    print("------------")
    g.close()
    return g

#from gutenberg.acquire import load_etext
#from gutenberg.cleanup import strip_headers
#print(gutenberg.acquire.text.local_path("data"))
#text = gutenberg.cleanup.strip_headers(gutenberg.acquire.load_etext(2701)).strip()
#print(text)  # prints 'MOBY DICK; OR THE WHALE\n\nBy Herman Melville ...'

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

def getMetadataForText(text_id):
    """
    Given a text ID number or rdf filename, return the metadata.
    """
    rdf = openRDF(text_id)
    
    #element_tree = None
    #ebook_metadata = ElementTree.parse(rdf)
    result = dict.fromkeys(META_FIELDS)
    result['id'] = getNumberFromFilename(text_id)
    
    # authors names
    creator_data = rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "creator"), None))
    authors_data = [rdf.triples((i[2], rdflib.term.URIRef(NS['pg'] + "name"), None)) for i in creator_data]
    authors_names = [j[2].toPython() for i in authors_data for j in i]
    result['author'] = authors_names
        
    # title
    result['title'] = [i[2].toPython() for i in rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "title"), None))][0]
    
    # LCC
    # languages
    languages_data = [rdf.triples((i[2], rdflib.term.URIRef(NS['rdf'] + "value"), None)) for i in rdf.triples( (None, rdflib.term.URIRef(NS['dc'] + "language"), None))]
    languages_values = [str(j[2].toPython()) for i in languages_data for j in i]
    result['language'] = languages_values
    # type
    # formats

    return result

print(getMetadataForText(18))
