Food Image to Recipe Application
==================================

This software allows you to upload a photo of a dish and receive a suggested recipe. It uses:
- A classification model for food detection in the image
- A language model for creating the recipe

The language model is also responsible for several features of the software:
 - Confirm the detected label is actually food
 - To suggest alternative dishes in cases where the image doesn't fully describe the food (e.g. green soup can be pees soup, brocoli soup or a combination of the two).

System Requirements:
---------------------
- Windows 10 or later
- No need for pre-installed Python or packages
- CPU-only execution (no GPU required)

How to Use:
------------
1. Unzip the software app and its dependencies
2. Launch the "Food Recipe App"
3. Choose your desired models
4. Upload an image (exemple exists in ./images)
5. Choose to generate descriptive suggestions or select the original detection
6. View and optionally save the generated recipe

Included Models:
-----------------
- MobileNetV2 and Swin-B (for image classification)
- GPT-2 Large amd InstructLM 1.3B (for text generation)
- Other suggested models can be used but requires internet connection

Sensitivities:
---------------
- Different classifier can be tested using check_classifier.py
- It appears that classifiers over the 22K ImageNet dataset excell over others.
- Classifiers that were fine-tuned on Food101 dataset better at food detection if exists, but result in larger number of false-positive predictions.

- Different language models can be tested using check_llm.py
    - Different prompts can be checked as well.
    - Changing the prompts for food existence confirmation and suggesting alternative dishes highly affect the results.
- Chat language models achieve better results in food existence confirmation and suggesting alternative dishes.
    - However, they are much heavier and slower in inference.
    - Moreover, changing the prompts
- SOTA models are using quantization and different input types, such as FP16 and BF16, for faster inference and reduced memory consumption.
    - However, these data types are not supported on CPU inference using Pytorch.

For Developers:
---------------
- To build from source, see `requirements.txt`
- To add or replace models, use `bundle_models.py`
- To freeze the app, use `pyinstaller pyinstaller.spec`
- To create a zipped package, run `zip_package.py`

License:
--------
MIT License (see LICENSE.txt if available)

Author:
--------
Stav Bar-Sheshet, March 2025


