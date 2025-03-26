import os
import re
import pandas as pd
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Parser:

    def __init__(self, chunk_size=4096, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            length_function = len,
        )
    
    def parse_file(self, file):
        reader = PdfReader(file)
        n_pages = len(reader.pages)

        filename = os.path.basename(file).split('.')[0]
        contents = self.get_table_of_contents(file)
        
        system = contents.pop(0)
        P = ''
        H = ''
        SH = ''
        C = ''

        n_rows = len(contents)

        row_is_header = [self._is_header(row) for row in contents]
        row_is_bigheader = self._is_bigheader(row_is_header)
        row_is_chapter = [self._is_chapter(row) for row in contents]
        row_page_n = [self._get_page_n(row) for row in contents]

        res_df = pd.DataFrame()
        i = 0

        while i < n_rows:
            row = contents[i]

            # Detect header
            if row.isupper() and not '.' in row:
                P = row
                i += 1
                continue
            else:
                row = self._clear_row(row)

            if row_is_header[i]:
                if row_is_header[i+1]:
                    H = row
                else:
                    SH = row
                i += 1
                continue

            C = row
            start_page = row_page_n[i]
            if i == n_rows-1:
                end_page = n_pages
            elif row_is_bigheader[i+1]:
                end_page = row_page_n[i+1] - 1
            elif row_is_header[i+1]:
                end_page = row_page_n[i+1]
            elif row_is_chapter[i+1]:
                end_page = row_page_n[i+1]
            else:
                end_page = row_page_n[i+2] - 1
            
            text = '\n\n'.join([reader.pages[n].extract_text() for n in range(start_page-1, end_page)])
            if len(text) > self.chunk_size:
                text = self.text_splitter.split_text(text)
            else:
                text = [text]
            n_texts = len(text)

            res_df = pd.concat([
                res_df, 
                pd.DataFrame({
                    'system': [system]*n_texts,
                    'part': [P]*n_texts,
                    'header': [H]*n_texts,
                    'subheader': [SH]*n_texts,
                    'chapter': [C]*n_texts,
                    'start_page': [start_page]*n_texts,
                    'end_page': [end_page]*n_texts,
                    'text': text
                })
            ]).reset_index(drop=True)

            i += 1
        res_df['id'] = [f'{filename}-{n}' for n in range(1, len(res_df)+1)]
        return res_df


    def get_table_of_contents(self, file):
        filename = os.path.basename(file).split('.')[0]

        reader = PdfReader(file)
        total_pages = len(reader.pages)
        
        contents = []
        for i in range(total_pages):
            page_text = reader.pages[i].extract_text()
            if '....' in page_text:
                contents += page_text.split('\n')
                page_name = f'{filename}-{i+1}'
                if page_name in contents:
                    contents.remove(page_name)
            else:
                break

        contents = contents[contents.index('CONTENTS')+1:]

        res_contents = [contents[0]]
        i = 1
        while i < len(contents):
            row = contents[i]

            if not '.' in row and len(row) < 30:
                res_contents.append(row)
                i += 1
                continue
            
            while not '.' in row:
                if row[-1] == '-':
                    row = row[:-1]
                i += 1
                row += contents[i]
            else:
                res_contents.append(row)
                i += 1
                continue
        
        res_contents = self._remove_excess_headers(res_contents)
                
        return res_contents
    
    def _remove_excess_headers(self, contents):
        res_contents = []
        c_header = [self._is_header(row) for row in contents]
        c_page_n = [self._get_page_n(row) for row in contents]

        for i in range(len(contents) - 2):
            if (c_header[i] and c_header[i+1] and c_header[i+2] and
                c_page_n == c_page_n == c_page_n):
                continue
            else:
                res_contents.append(contents[i])
        res_contents = res_contents + contents[-2:]
        return res_contents
    
    def _get_page_n(self, text):
        numbers = re.findall(r'\d+', text)
        if len(numbers) > 0:
            return int(numbers[-1])
        else:
            return -1
    
    def _is_chapter(self, text):
        if not text.isupper() and '.' in text:
            return True
        else:
            return False
    
    def _is_bigheader(self, row_is_header):
        row_is_bigheader = [True] + [row_is_header[i] and row_is_header[i+1] for i in range(1, len(row_is_header))]
        return row_is_bigheader
    
    def _is_header(self, text):
        if text.isupper() and '.' in text:
            return True
        else:
            return False
    
    def _clear_row(self, text):
        text = text[:text.index('.')].strip()
        return text
