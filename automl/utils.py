"""
**utils** module provide a set of generic utility functions and decorators, which are not data-related
(data manipulation utility functions should go to **datautils**).

important
    This module can be imported by any other module (especially framework integration modules),
    therefore, it should have as few external dependencies as possible,
    and should have no dependency to any other **automl** module.
"""
import datetime as dt
from functools import reduce, wraps
import json
import logging
import os
import shutil
import stat
import sys
import time

import psutil
from ruamel import yaml

try:
    from pip._internal import main as pip_main
except ImportError:
    from pip import main as pip_main


log = logging.getLogger(__name__)


class Namespace:

    mangled_prefix = '_Namespace__'

    @staticmethod
    def merge(*namespaces, deep=False):
        merged = Namespace()
        for ns in namespaces:
            if ns is None:
                continue
            if not deep:
                merged + ns
            else:
                for k, v in ns:
                    if isinstance(v, Namespace):
                        merged[k] = Namespace.merge(merged[k], v, deep=True)
                    else:
                        merged[k] = v
        return merged

    @staticmethod
    def dict(namespace):
        dic = dict(namespace)
        for k, v in dic.items():
            if isinstance(v, Namespace):
                dic[k] = Namespace.dict(v)
        return dic

    def __init__(self, *args, **kwargs):
        self.__ns = dict(*args, **kwargs)

    def __add__(self, other):
        """extends self with other (always overrides)"""
        self.__ns.update(other)
        return self

    def __mod__(self, other):
        """extends self with other (adds only missing keys)"""
        for k, v in other:
            self.__ns.setdefault(k, v)
        return self

    def __contains__(self, key):
        return key in self.__ns

    def __len__(self):
        return len(self.__ns)

    def __getattr__(self, name):
        if name.startswith(Namespace.mangled_prefix):
            return super().__getattr__(name)
        elif name in self.__ns:
            return self.__ns[name]
        raise AttributeError(name)

    def __setattr__(self, key, value):
        if key.startswith(Namespace.mangled_prefix):
            super().__setattr__(key, value)
        else:
            self.__ns[key] = value

    def __getitem__(self, item):
        return self.__ns.get(item)

    def __setitem__(self, key, value):
        self.__ns[key] = value

    def __iter__(self):
        return iter(self.__ns.items())

    def __copy__(self):
        return Namespace(self.__ns.copy())

    def __dir__(self):
        return list(self.__ns.keys())

    def __str__(self):
        return str(self.__ns)

    def __repr__(self):
        return repr(self.__ns)


def repr_def(obj):
    return "{clazz}({attributes})".format(clazz=type(obj).__name__, attributes=', '.join(("{}={}".format(k, repr(v)) for k, v in obj.__dict__.items())))


_CACHE_PROP_PREFIX_ = '__cached__'


def _cached_property_name(fn):
    return _CACHE_PROP_PREFIX_ + (fn.__name__ if hasattr(fn, '__name__') else str(fn))


def clear_cache(self, functions=None):
    cached_properties = [prop for prop in dir(self) if prop.startswith(_CACHE_PROP_PREFIX_)]
    properties_to_clear = cached_properties if functions is None \
        else [prop for prop in [_cached_property_name(fn) for fn in functions] if prop in cached_properties]
    for prop in properties_to_clear:
        delattr(self, prop)
    log.debug("Cleared cached properties: %s", properties_to_clear)


def cache(self, key, fn):
    """

    :param self: the object that will hold the cached value
    :param key: the key/attribute for the cached value
    :param fn: the function returning the value to be cached
    :return: the value returned by fn on first call
    """
    if not hasattr(self, key):
        value = fn(self)
        setattr(self, key, value)
    return getattr(self, key)


def cached(fn):
    """

    :param fn:
    :return:
    """
    result = _cached_property_name(fn)

    def decorator(self):
        return cache(self, result, fn)

    return decorator


def memoize(fn):
    prop_name = _cached_property_name(fn)

    def decorator(self, key=None):  # TODO: could support unlimited args by making a tuple out of *args + **kwargs: not needed for now
        memo = cache(self, prop_name, lambda _: {})
        if not isinstance(key, str) and hasattr(key, '__iter__'):
            key = tuple(key)
        if key not in memo:
            memo[key] = fn(self) if key is None else fn(self, key)
        return memo[key]

    return decorator


def lazy_property(prop_fn):
    """

    :param prop_fn:
    :return:
    """
    prop_name = _cached_property_name(prop_fn)

    @property
    def decorator(self):
        return cache(self, prop_name, prop_fn)

    return decorator


def profile(logger=log, log_level=None, duration=True, memory=True):
    ps = psutil.Process() if memory else None

    def decorator(fn):

        @wraps(fn)
        def profiler(*args, **kwargs):
            nonlocal log_level
            log_level = log_level or (logging.TRACE if hasattr(logging, 'TRACE') else logging.DEBUG)
            if not logger.isEnabledFor(log_level):
                return fn(*args, **kwargs)

            before_mem = ps.memory_full_info() if memory else 0
            start = time.time() if duration else 0
            ret = fn(*args, **kwargs)
            stop = time.time() if duration else 0
            after_mem = ps.memory_full_info() if memory else 0
            name = fn_name(fn)
            if duration:
                logger.log(log_level, "[PROFILING] `%s` executed in %.3fs", name, stop-start)
            if memory:
                ret_size = obj_size(ret)
                if ret_size > 0:
                    logger.log(log_level, "[PROFILING] `%s` returned object size: %.3f MB", name, to_mb(ret_size))
                logger.log(log_level, "[PROFILING] `%s` memory change; process: %+.2f MB/%.2f MB, resident: %+.2f MB/%.2f MB, virtual: %+.2f MB/%.2f MB",
                           name,
                           to_mb(after_mem.uss-before_mem.uss),
                           to_mb(after_mem.uss),
                           to_mb(after_mem.rss-before_mem.rss),
                           to_mb(after_mem.rss),
                           to_mb(after_mem.vms-before_mem.vms),
                           to_mb(after_mem.vms))
            return ret

        return profiler

    return decorator


def fn_name(fn):
    return ".".join([fn.__module__, fn.__qualname__])


def obj_size(o):
    if o is None:
        return 0
    # handling numpy obj size (nbytes property)
    return o.nbytes if hasattr(o, 'nbytes') else sys.getsizeof(o, -1)


def to_mb(size_in_bytes):
    return size_in_bytes / (1 << 20)


def flatten(iterable, flatten_tuple=False, flatten_dict=False):
    return reduce(lambda l, r: (l.extend(r) if isinstance(r, (list, tuple) if flatten_tuple else list)
                                else l.extend(r.items()) if flatten_dict and isinstance(r, dict)
                                else l.append(r)) or l, iterable, [])


class YAMLNamespaceLoader(yaml.loader.SafeLoader):

    @classmethod
    def init(cls):
        cls.add_constructor(u'tag:yaml.org,2002:map', cls.construct_yaml_map)

    def construct_yaml_map(self, node):
        data = Namespace()
        yield data
        value = self.construct_mapping(node)
        data + value


YAMLNamespaceLoader.init()


def json_load(file, as_namespace=False):
    if as_namespace:
        return json.load(file, object_hook=lambda dic: Namespace(**dic))
    else:
        return json.load(file)


def yaml_load(file, as_namespace=False):
    if as_namespace:
        return yaml.load(file, Loader=YAMLNamespaceLoader)
    else:
        return yaml.safe_load(file)


def config_load(path, verbose=False):
    path = normalize_path(path)
    if not os.path.isfile(path):
        log.log(logging.WARNING if verbose else logging.DEBUG, "No config file at `%s`, ignoring it.", path)
        return Namespace()

    _, ext = os.path.splitext(path.lower())
    loader = json_load if ext == 'json' else yaml_load
    log.log(logging.INFO if verbose else logging.DEBUG, "Loading config file `%s`.", path)
    with open(path, 'r') as file:
        return loader(file, as_namespace=True)


def datetime_iso(datetime=None, date=True, time=True, micros=False, date_sep='-', datetime_sep='T', time_sep=':', micros_sep='.', no_sep=False):
    """

    :param date:
    :param time:
    :param micros:
    :param date_sep:
    :param time_sep:
    :param datetime_sep:
    :param micros_sep:
    :param no_sep: if True then all separators are taken as empty string
    :return:
    """
    # strf = "%Y{ds}%m{ds}%d{dts}%H{ts}%M{ts}%S{ms}%f".format(ds=date_sep, ts=time_sep, dts=datetime_sep, ms=micros_sep)
    if no_sep:
        date_sep = time_sep = datetime_sep = micros_sep = ''
    strf = ""
    if date:
        strf += "%Y{_}%m{_}%d".format(_=date_sep)
        if time:
            strf += datetime_sep
    if time:
        strf += "%H{_}%M{_}%S".format(_=time_sep)
        if micros:
            strf += "{_}%f".format(_=micros_sep)
    datetime = dt.datetime.utcnow() if datetime is None else datetime
    return datetime.strftime(strf)


def str2bool(s):
    if s.lower() in ('true', 't', 'yes', 'y', 'on', '1'):
        return True
    elif s.lower() in ('false', 'f', 'no', 'n', 'off', '0'):
        return False
    else:
        raise ValueError(s+" can't be interpreted as a boolean")


def str_def(s, if_none=''):
    if s is None:
        return if_none
    return str(s)


def head(s, lines=10):
    s_lines = s.splitlines() if s else []
    return '\n'.join(s_lines[:lines])


def tail(s, lines=10, from_line=None, include_line=True):
    if s is None:
        return None if from_line is None else None, None

    s_lines = s.splitlines()
    start = -lines
    if isinstance(from_line, int):
        start = from_line
        if not include_line:
            start += 1
    elif isinstance(from_line, str):
        try:
            start = s_lines.index(from_line)
            if not include_line:
                start += 1
        except ValueError:
            start = 0
    last_line = dict(index=len(s_lines) - 1,
                     line=s_lines[-1] if len(s_lines) > 0 else None)
    t = '\n'.join(s_lines[start:])
    return t if from_line is None else (t, last_line)


def pip_install(module_or_requirements, is_requirements=False):
    try:
        if is_requirements:
            pip_main(['install', '--no-cache-dir', '-r', module_or_requirements])
        else:
            pip_main(['install', '--no-cache-dir', module_or_requirements])
    except SystemExit as se:
        log.error("error when trying to install python modules %s", module_or_requirements)
        log.exception(se)


def normalize_path(path):
    return os.path.realpath(os.path.expanduser(path))


def split_path(path):
    dir, file = os.path.split(path)
    base, ext = os.path.splitext(file)
    return Namespace(dirname=dir, filename=file, basename=base, extension=ext)


def path_from_split(split, real_path=True):
    return os.path.join(os.path.realpath(split.dirname) if real_path else split.dirname,
                        split.basename)+split.extension


def dir_of(caller_file, rel_to_project_root=False):
    abs_path = os.path.dirname(os.path.realpath(caller_file))
    if rel_to_project_root:
        project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        return os.path.relpath(abs_path, project_root)
    else:
        return abs_path


def touch(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a'):
        os.utime(file_path, times=None)


def backup_file(file_path):
    src_path = os.path.realpath(file_path)
    if not os.path.isfile(src_path):
        return
    p = split_path(src_path)
    mod_time = dt.datetime.utcfromtimestamp(os.path.getmtime(src_path))
    dest_name = ''.join([p.basename, '_', datetime_iso(mod_time, date_sep='', time_sep=''), p.extension])
    dest_dir = os.path.join(p.dirname, 'backup')
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, dest_name)
    shutil.copyfile(src_path, dest_path)
    log.debug('File `%s` was backed up to `%s`.', src_path, dest_path)


def run_cmd(cmd, return_output=True, *args, **kvargs):
    # TODO: switch to subprocess module (Popen) instead of os? would allow to use timeouts and kill signal
    #   besides, this implementation doesn't seem to work well with some commands if output is not read.
    output = None
    cmd_args = list(filter(None, []
                                 + ([] if args is None else list(args))
                                 + flatten(kvargs.items(), flatten_tuple=True) if kvargs is not None else []
                           ))
    full_cmd = ' '.join([cmd]+cmd_args)
    log.info("Running cmd `%s`.", full_cmd)
    with os.popen(full_cmd) as subp:
        if return_output:
            output = subp.read()
    if subp.close():
        log.debug(output)
        output_tail = tail(output, 25) if output else 'Unknown Error'
        raise OSError("Error when running command `{cmd}`: {error}".format(cmd=full_cmd, error=output_tail))
    return output


def call_script_in_same_dir(caller_file, script_file, *args, **kvargs):
    here = dir_of(caller_file)
    script = os.path.join(here, script_file)
    mod = os.stat(script).st_mode
    os.chmod(script, mod | stat.S_IEXEC)
    output = run_cmd(script, True, *args, **kvargs)
    log.debug(output)
    return output


def system_memory_mb():
    vm = psutil.virtual_memory()
    return Namespace(
        total=to_mb(vm.total),
        available=to_mb(vm.available)
    )


def system_cores():
    return psutil.cpu_count()
