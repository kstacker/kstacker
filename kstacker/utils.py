import os
import shutil


def get_path(params, key):
    return os.path.join(os.path.expanduser(params["work_dir"]), params[key])


def create_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
