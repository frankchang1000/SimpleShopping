# SimplisticShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license : Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Finds recipe information and food information from a barcode or 
   image anaylsis scan."""
# ---------------------------------------------------------------------------

import fatsecret

from . import file_reader


class NutritionInfo:
    def __init__(self, 
                 consumer_key: str,
                 consumer_secret: str,
                 max_recipes: int = 10):
        self.fs = fatsecret.Fatsecret(consumer_key, consumer_secret)
        self.max_recipes = max_recipes

    def find_recipes(self,
                     items: list) -> list:
        """Finds recipes from items.
        Params:
            fs: fatsecret.Fatsecret
                The fatsecret client.
            items: list
                The items to search through.
        Returns:
            recipes: list
                The list of dict recipes.
        """
        recipes = []
        search_query = " ".join(items)
        results = self.fs.recipes_search(
            search_expression=search_query)
        if self.max_recipes == 0:
            self.max_recipes = 1
        for i, recipe in enumerate(results):
            if i >= self.max_recipes:
                break
            recipes.append(
                {"recipe_name": recipe["recipe_name"],
                 "recipe_description": recipe["recipe_description"],
                 "calories_per_serving": recipe["recipe_nutrition"]["calories"],
                 "fat_per_serving": recipe["recipe_nutrition"]["fat"],
                 "carbohydrate_per_serving": recipe["recipe_nutrition"]["carbohydrate"],
                 "protein_per_serving": recipe["recipe_nutrition"]["protein"]})
        return recipes

    def nutritional_info(self,
                         item: str) -> dict:
        """Finds nutritional information from item.
        Params:
            fs: fatsecret.Fatsecret
                The fatsecret client.
            item: str
                The item to search for.
        Returns:
            nutritional_info: dict
                The dict of nutritional information.
        """
        results = self.fs.foods_search(
            search_expression=item)
        food = results[0]
        return {
            "food_name": food["food_name"],
            "food_description": food["food_description"]}

    def find_foods(self,
                   items: list) -> list:
        """Finds foods from items.
        Params:
            fs: fatsecret.Fatsecret
                The fatsecret client.
            items: list
                The items to search through.
        Returns:
            foods: list
                The list of dict foods.
        """
        foods = []
        for item in items:
            foods.append(self.nutritional_info(item))
        return foods
