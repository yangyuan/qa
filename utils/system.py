import os
import shutil


def ensure_dir(folder):
    if isinstance(folder, str):
        if not os.path.exists(folder):
            os.makedirs(folder)
        return

    if isinstance(folder, list):
        for _folder in folder:
            ensure_dir(_folder)
        return

    raise Exception()


def remove_dir(folder):
    if isinstance(folder, str):
        shutil.rmtree(folder, ignore_errors=True)
        return

    if isinstance(folder, list):
        for _folder in folder:
            remove_dir(_folder)
        return

    raise Exception()
