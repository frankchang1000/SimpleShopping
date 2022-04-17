# SimpleShopping

Grocery shopping made convenient.

<p align="center">
  <img src="https://github.com/frankchang1000/SimpleShopping/blob/main/data/logo.png", width="250"/>
</p>

## What is SimpleShopping?

<p align="center">
  <img src="https://github.com/frankchang1000/SimpleShopping/blob/main/data/SimpleShopping-functions.png", width=450/>
</p>

## A Novel AI Implementation

### Overview

SimpleShopping uses the novel semi-supervised training method: Advanced Meta Pseudo Labels. Advanced Meta Pseudo Labels uses AI to create new efficient AI models.

Compared to previous methods, SimpleShopping is 10% more accurate than state of the art attempts.

### Dataset

We used the [VegFru](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hou_VegFru_A_Domain-Specific_ICCV_2017_paper.pdf) dataset which contains over 80000 images of vegetables and fruits.

<p align="center">
  <img src="https://github.com/frankchang1000/SimpleShopping/blob/main/data/SimpleShopping-dataset.png", width="450"/>
</p>

## Model Architecture

A modified EffcientNet Convolutional Neural Netwokrk (CNN) model was used to make inferences.

EfficientNet is a SOTA model developed by Google in 2020. For more information, please refer to [EfficientNet](https://arxiv.org/abs/1905.11946).

## Github Copilot: Assisting the Open Source Community

We utilized Github copilot: a platform that enables efficient coding. Copilot helps lint and write code with a complex JPT model developed by OpenAI. It is currently in development beta, and we thank Github and OpenAI for the opportunity to test their new product.

### Clean GitHub

Our [GitHub](https://github.com/frankchang1000/SimpleShopping/blob/main/README.md) was cleanly created and with accurate PEP8 formatting. The code is ready for use with its easy to read API and well documented components.

## APIs Used

To create simple shopping, we utilized three core APIs: Tensorflow, FatSecret, and Streamlit. We utilized Tensorflow to create our neural network models and to process data. We employed mixed precision training, which increased performance while also reducing computational power, and we used JIT compilation for our models.

FatSecret is a powerful API used by many large - and small - companies around the world, notably including Samsung, Amazon, and Fitbit. We used the FatSecret API to access and and find nutrition info, recipes, and other important information for our application.

Lastly, we used Streamlit to deploy our novel algorithms into a useraccessible manner. 

### Issues with API and Development

Throughout the hackathon, numerous issues arose from incompatibility between the APIs used (Tensorflow, OpenCV, ...) however, by implementing a common medium of array transfers - Numpy - we were able to resolve the issues. Furthermore, we faced numerous issues with the frontend API: Streamlit, but we were able to resolves those issues using a pseudo-HTML based formatting.

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

Then place your OAuth1 token into a fat_secret.key file in the data folder.

```
{
    "consumer_key": "abunchofrandomchars",
    "consumer_secret": "abunchofrandomchars"
}
```

## Running the GUI Application

To run the application, run the following command in the project directory:

```python
streamlit run main.py
```

To finetune training or to run training:

```python
python main_tf2.py 
```

## Refernces

[1] Tan, M., & Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning (pp. 6105-6114). PMLR.

[2] Hou, S., Feng, Y., & Wang, Z. (2017). Vegfru: A domain-specific dataset for fine-grained visual categorization. In Proceedings of the IEEE International Conference on Computer Vision (pp. 541-549).

[3] Pham, H., Dai, Z., Xie, Q., & Le, Q. V. (2021). Meta pseudo labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 11557-11568).

## License

Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (CC BY-NC-ND 4.0)


    Under the following terms:

    Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

    NonCommercial — You may not use the material for commercial purposes.

    NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.

    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    SimpleShopping Team 2022: Frank Chang (FrankChang1000), Thomas Chia (IdeaKing), Alton Lin (alton.d.lin), and Jason Yoo (youajason)
