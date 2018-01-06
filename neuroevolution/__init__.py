import os
import shutil
from io import BytesIO
from chainer import serializers
from minio import Minio
from minio.error import BucketAlreadyOwnedByYou


class MinioModelStore(object):

    def __init__(self, access_key, secret_key):
        self.bucket = 'neuroevolution'
        self.access_key = access_key
        self.secret_key = secret_key

    def _get_client(self):
        client = Minio(
            'minio:9000', self.access_key, self.secret_key, secure=False)
        return client

    def save(self, name, model):
        client = self._get_client()
        try:
            client.make_bucket(self.bucket)
        except BucketAlreadyOwnedByYou:
            pass
        fname = 'model/{}.npz'.format(name)
        fp = BytesIO()
        serializers.save_npz(fp, model)
        length = len(fp.getvalue())
        fp.seek(0)
        client.put_object(self.bucket, fname, fp, length)

    def load(self, name, model):
        client = self._get_client()
        fname = 'model/{}.npz'.format(name)
        resp = client.get_object(self.bucket, fname)
        serializers.load_npz(BytesIO(resp.data), model)
        return model
