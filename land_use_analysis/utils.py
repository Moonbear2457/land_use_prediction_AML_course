# land_use_analysis/utils.py

import os
from typing import List
import logging

logger = logging.getLogger(__name__)

def extract_years(file_paths: List[str]) -> List[int]:
    years = []
    for file_path in file_paths:
        try:
            parts = os.path.basename(file_path).split('_')
            year = int(parts[1].split('.')[0])
            years.append(year)
        except (IndexError, ValueError):
            logger.warning(f"Could not extract year from '{file_path}'")
    return years

