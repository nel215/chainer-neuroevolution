import os
import shutil
from chainer import serializers



class ModelStore(object):

    def __init__(self):
        self.dir = '/data'

    def save(self, name, model):
        fname = '{}.npz'.format(name)
        path = os.path.join(self.dir, fname)
        serializers.save_npz(path, model)

    def load(self, name, model):
        fname = '{}.npz'.format(name)
        path = os.path.join(self.dir, fname)
        serializers.load_npz(path, model)
        return model
