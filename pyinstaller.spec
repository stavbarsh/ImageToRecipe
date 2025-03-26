# --- food_image_to_recipe.spec ---
# Build with: pyinstaller pyinstaller.spec

block_cipher = None

import os
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('torch') + collect_submodules('torchvision') + collect_submodules('transformers') + collect_submodules('tkinter')


a = Analysis(
    ['food_image_to_recipe.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FoodRecipeApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True  # Set to True if you want to see terminal output
)
