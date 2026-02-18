# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files
from PyInstaller.building.build_main import Tree
import os

datas = []
binaries = []
hiddenimports = []

# Add assets folder (images and icons) using Tree
assets_tree = Tree('assets', prefix='assets')

# Collect PyDrive
hiddenimports += collect_submodules('pydrive')
tmp_ret = collect_data_files('pydrive', include_py_files=True)
datas += tmp_ret

# Collect oauth2client (required by PyDrive)
hiddenimports += collect_submodules('oauth2client')

# Collect Google API dependencies
hiddenimports += collect_submodules('googleapiclient')
hiddenimports += collect_submodules('google_auth_oauthlib')


a = Analysis(
    ['cloud_chatbot.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cloud_chatbot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets\\kitview_icon.png'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    assets_tree,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cloud_chatbot',
)
