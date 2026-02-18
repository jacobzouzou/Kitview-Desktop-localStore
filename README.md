# README
## Compile insttructions
instutions de compilation: 
pyinstaller --onefile --windowed yourPyFile.py
or
pyinstaller --onedir --windowed yourPyFile.py


## Install local RAG
pip install faiss-cpu sentence-transformers numpy
pip install pypdf python-docx python-pptx beautifulsoup4 openpyxl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
