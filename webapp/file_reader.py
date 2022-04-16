# SimplisticShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Util functions for the webapp."""
# ---------------------------------------------------------------------------

import os
import json
import csv


def read_keys(
    file_path: str = "data/fat_secret.key") -> dict:
    """Reads a JSON file.
    Params:
        file_path: str
            The file path to the JSON file.
    Returns:
        keys: dict
            The keys.
    """
    with open(file_path, "r") as f:
        keys = json.load(f)
    return keys


def read_items_file(
    file_path: str = "data/upc_corpus.csv") -> dict:
    """Reads an image from a file.
    Params:
        file_path: str
            The file path to UPC file.
    Returns:
        items: dict
            The items.
    """
    items = {}
    if os.path.isfile(file_path):
        with open("data/upc_corpus.csv", mode="r", encoding="utf8") as bc_csv:
            reader = csv.reader(bc_csv)
            items = {rows[0]:rows[1] for rows in reader}
    return items
