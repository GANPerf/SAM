import os
from typing import List
from torchvision.datasets.utils import download_and_extract_archive


def download_data(root: str, file_name: str, archive_name: str, url_link: str):

    if not os.path.exists(os.path.join(root, file_name)):
        print("Downloading {}".format(file_name))
        # if os.path.exists(os.path.join(root, archive_name)):
        #     os.remove(os.path.join(root, archive_name))
        try:
            download_and_extract_archive(url_link, download_root=root, filename=archive_name, remove_finished=False)
        except Exception:
            print("Fail to download {} from url link {}".format(archive_name, url_link))
            print('Please check you internet connection or '
                  "reinstall DALIB by 'pip install --upgrade dalib'")
            exit(0)


def check_exits(root: str, file_name: str):
    """Check whether `file_name` exists under directory `root`. """
    if not os.path.exists(os.path.join(root, file_name)):
        print("Dataset directory {} not found under {}".format(file_name, root))
        exit(-1)


def read_list_from_file(file_name: str) -> List[str]:
    """Read data from file and convert each line into an element in the list"""
    result = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            result.append(line.strip())
    return result

def save_list_to_file(file_name: str, file_list: List[str]):
    textfile = open(file_name, "w")
    for element in file_list:
        textfile.write(element + "\n")
    textfile.close()