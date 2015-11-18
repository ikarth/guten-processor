import os
import dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load_dotenv(dotenv_path)

# Grab environment variables...
FILEPATH_RDF = os.environ.get("GUTENBERG_RDF_FILEPATH")
FILEPATH_GUTENBERG_TEXTS = os.environ.get("GUTENBERG_DATA")

LIVE_FILE_LIST = []
with open(FILEPATH_GUTENBERG_TEXTS + os.sep + "current_local_text_files.txt", encoding='utf-8') as file:
    for line in file:
        LIVE_FILE_LIST.append(int(str(line.rstrip())))
    
