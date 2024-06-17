"""
provide Cache class
"""
import functools
import inspect
import os
import pickle
import warnings
import shutil

_tmpprefix = '.tmp'

class Cache(object):
    """
    class to mix into other class for implementing file system based cache.
    class must have 'path' attribute, that point a directory to save data files.

    get_cache(name) get cache.

    when pickled, __getstate__ does dumping each attribute beginning with
    '_data_' into files.

    TODO: file path seems absolute, which is not good
    """
    def get_cache(self, name):
        if hasattr(self, '_data_' + name):
            return getattr(self, '_data_' + name)
        elif hasattr(self, '_path_' + name):
            path = getattr(self, '_path_' + name)
            with open(path, 'rb') as f:
                data = pickle.load(f)
                setattr(self, '_data_' + name, data)
            return data
        else:
            func = getattr(self, name)
            orig_func = func.orig_func
            argspec = inspect.getfullargspec(orig_func)
            # print argspec
            args, varargs, kws, defaults, *_ = argspec
            # print argspec
            if args == ['self'] and varargs is None and kws is None:
                return func()
            elif defaults is not None and varargs is None and kws is None \
                and len(args) - 1 == len(defaults):
                warnings.warn('calculating %s() with default arguments...' % orig_func.func_name)
                return func()
            else:
                raise RuntimeError('No cache for %s.%s!' % (type(self).__name__, name))
    def has_cache(self, name):
        return hasattr(self, '_data_' + name) or hasattr(self, '_path_' + name)
    def set_cache(self, name, value):
        setattr(self, '_data_' + name, value)
    def remove_cache(self, name):
        if hasattr(self, '_data_' + name):
            delattr(self, '_data_' + name)
        if hasattr(self, '_path_' + name):
            delattr(self, '_path_' + name)
    def _dump_data_to_path(self, name):
        if hasattr(self, '_data_' + name):
            data = getattr(self, '_data_' + name)
            path = os.path.join(self.path, name)
            setattr(self, '_path_' + name, path)
            with open(path + _tmpprefix, 'wb') as f:
                pickle.dump(data, f, -1)
            shutil.move(path + _tmpprefix, path)
    def __getstate__(self):
        ## save heavy data to file
        for k in list(self.__dict__.keys()):
            if k[:6] == '_data_':
                self._dump_data_to_path(k[6:])
        ## delete heavy objects before pickling,
        ##  which can be reconstructed from path or something
        dic = self.__dict__.copy()
        for k in list(self.__dict__.keys()):
            if k[:6] == '_data_':
                del dic[k]
        return dic
    def save(self, path=None, clear_memory=False):
        if not path and self.path:
            path = self.path
        filepath = os.path.join(path, 'self')
        with open(filepath + _tmpprefix, 'wb') as f:
            tmp = self.path
            self.path = path
            pickle.dump(self, f, -1)
            self.path = tmp
        shutil.move(filepath + _tmpprefix, filepath)
        if clear_memory:
            for k, v in self.__dict__.copy().items():
                if k[:6] == '_data_':
                    del self.__dict__[k]
    @classmethod
    def load(cls, path):
        filepath = os.path.join(path, 'self')
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class do_cache(object):
    """
    to be used as decorator Cache (-inherited) class.
    the most recent result for that function is stored as `.cache`.

    $ cat foo.py
    from cache import Cache, do_cache
    from some_module import long_calculation_and_heavy_data

    class Foo(Cache):
       def __init__(self, path):
           self.path = path
       @do_cache
       def func(self, args)
           return long_calculation_and_heavy_data()
    $ python
    >>> import foo
    >>> foo = Foo('/tmp/foo')
    >>> print foo.func(arg)          # do calculation
    <something>
    >>> print foo.func.cache         # now the result is also stored in cache
    <something>
    >>> foo.save()
    >>> quit()
    $ python
    >>> import foo # class definition is needed for loading
    >>> from cache import Cache
    >>> foo = Cache.load('/tmp/foo') # or foo.Foo.load('/tmp/foo')
    >>> print foo.func.cache         # load previous result from cache
    <something>
    """
    # https://christiankaula.com/python-decorate-method-gets-class-instance.html
    def __init__(self, func):
        self.orig_func = func

    def __call__(self, *args, **kwargs):
        result = self.orig_func(self._obj, *args, **kwargs)
        name = self.orig_func.__name__
        setattr(self._obj, '_data_' + name, result)
        return result

    @property
    def cache(self):
        return self._obj.get_cache(self.orig_func.__name__)

    def __get__(self, instance, owner):
        self._cls = owner
        self._obj = instance

        return self
