import mimetypes
import boto3
import os
from tqdm import tqdm
import sys


if len(sys.argv) < 2:
    raise ValueError("Need to provide the Bucket Name")


FLAMBE_BUCKET_NAME = sys.argv[1]


def get_flambe_bucket(s3):
    for b in s3.buckets.all():
        if b.name == FLAMBE_BUCKET_NAME:
            return b

    raise Exception("Flambe bucket not found")


def get_mime_type(path):
    mimetype, _ = mimetypes.guess_type(path)
    if mimetype is None:
        raise Exception("Failed to guess mimetype")
    return mimetype


def upload_documentation(bucket, doc_html_dir):
    # enumerate local files recursively
    for root, dirs, files in tqdm(os.walk(doc_html_dir)):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full path
            s3_path = os.path.relpath(local_path, doc_html_dir)
            try:
                bucket.upload_file(local_path, s3_path, ExtraArgs={
                    "ContentType": get_mime_type(local_path)
                })
            except Exception:
                print(f"Could not upload {s3_path}.")


if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    html_dir = os.path.join(_dir, "..", "docs", "_build", "html")

    if not os.path.exists(html_dir):
        print("Docuementation HTML not found. Were the docs built?")
        sys.exit(1)

    s3 = boto3.resource('s3')
    b = get_flambe_bucket(s3)
    upload_documentation(b, html_dir)
