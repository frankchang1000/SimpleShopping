# SimpleShopping

## Grocery shopping made convenient

<p align="center">
  <img src="https://github.com/frankchang1000/SimpleShopping/blob/main/data/logo.png">
</p>

## What is SimpleShopping?

<p align="center">
  <img src="https://github.com/frankchang1000/SimpleShopping/blob/main/data/SimpleShopping-functions.png">
</p>

## Installation

```python
git clone https://github.com/frankchang1000/SimpleShopping.git
cd simpleshopping
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Use 7zip to unzip the tensorflow model file. It is located in data/simpleshopping-model.7z

To access the nutrition data, you MUST have a developer Fatsecret API key. For more information about FatSecret API, please refer to [here](https://platform.fatsecret.com/api/Default.aspx).

Then place your OAuth1 token into a fat_scret.key file in the data folder.

```
{
    "consumer_key": "abunchofrandomchars",
    "consumer_secret": "abunchofrandomchars"
}
```