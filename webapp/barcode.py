# SimplisticShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Detects barcodes from an image and locates the product info."""
# ---------------------------------------------------------------------------

import numpy as np
import pyzbar.pyzbar as pb


def scan_barcode(
    input_image: np.array,
    debug: bool = False) -> str:
    """Reads barcodes from an image.
    Params:
        input_image: np.array
            The input image.
        debug: bool
            Whether to print debug information.
    Returns:
        barcode: str
            The barcode.
    """
    barcodes = pb.decode(input_image)
    if len(barcodes) > 0:
        barcode = barcodes[0].data.decode("utf-8")
    else:
        barcode = None
    if debug:
        print("Barcode: {}".format(barcode))
    return barcode


def find_product(
    barcode: str,
    items: dict) -> str:
    """Finds the product info from the barcode.
    Params:
        barcode: str
            The barcode.
        items: dict
            The items to search through.
    Returns:
        product: str
            The product info.
    """
    product = None
    if barcode is not None:
        try:
            product = items[barcode]
        except KeyError:
            product = str("Product not found.")
    return product
