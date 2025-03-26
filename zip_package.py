import os
import zipfile

OUTPUT_ZIP = "FoodRecipeApp_Package.zip"
FILES_TO_INCLUDE = [
    "FoodRecipeApp.exe",
    "README.txt",
    'food_image_to_recipe.py',
    'bundle_models.py',
    'check_classifier.py',
    'check_llm.py',
    'requirements.txt',
    'pyinstaller.spec',
    'zip_package.py'

]

DIRECTORIES_TO_INCLUDE = [
    "models", "images"
]

def count_files():
    count = len(FILES_TO_INCLUDE)
    for folder in DIRECTORIES_TO_INCLUDE:
        for _, _, files in os.walk(folder):
            count += len(files)
    return count

total = count_files()
done = 0

with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in FILES_TO_INCLUDE:
        arcname = os.path.basename(file)
        zipf.write(file, arcname)
        done += 1
        print(f"[{done}/{total}] Added: {file}")

    for folder in DIRECTORIES_TO_INCLUDE:
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=os.path.dirname(folder))
                zipf.write(full_path, arcname)
                done += 1
                print(f"[{done}/{total}] Added: {full_path}")

print(f"\nPackage created: {OUTPUT_ZIP}")
