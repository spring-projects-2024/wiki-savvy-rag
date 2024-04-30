#!/bin/bash

cd ~/path/to/project/wikipedia_data
python extract_category.py
python select_categories.py
python subsample_pages.py
python clean_dump.py
python remove_xml_tags.py
python chunk_data.py