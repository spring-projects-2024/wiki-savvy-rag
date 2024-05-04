#!/bin/bash

# This script should be run from the root directory of the project!
# It expects the path to the xml dump file as an argument.

DUMP_PATH=$1
DELETE_OLD_FILES=$2

python wikidump_processing/scripts/extract_category.py
python wikidump_processing/scripts/select_categories.py
python wikidump_processing/scripts/subsample_pages.py --dump_path $DUMP_PATH --delete_old_files $DELETE_OLD_FILES
python wikidump_processing/scripts/clean_dump.py --delete_old_files $DELETE_OLD_FILES
python wikidump_processing/scripts/remove_xml_tags.py --delete_old_files $DELETE_OLD_FILES
python wikidump_processing/scripts/chunk_data.py --delete_old_files $DELETE_OLD_FILES
python -m spacy download en_core_web_sm
python wikidump_processing/scripts/split_long_chunks.py --delete_old_files $DELETE_OLD_FILES
