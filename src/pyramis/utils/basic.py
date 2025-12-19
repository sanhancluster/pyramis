from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .. import config
import multiprocessing as mp
import time
import sys

DEFAULT_N_PROCS = config['DEFAULT_N_PROCS']

def in_jupyter():
    try:
        from IPython.core.getipython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except:
        return False


def get_mp_context():
    if in_jupyter():
        return mp.get_context("spawn")
    try:
        return mp.get_context("forkserver")
    except ValueError:
        return mp.get_context("spawn")


def get_mp_executor(backend: str="thread", n_workers: int=DEFAULT_N_PROCS):
    ctx = get_mp_context()

    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    executor_kwargs: dict = {'max_workers': n_workers}
    if backend == "process":
        executor_kwargs['mp_context'] = ctx
    return Executor(**executor_kwargs)


CYAN = "\033[36m"
GREEN = "\033[33m"
RESET = "\033[0m"
class Timestamp:
    """
    A class to export time that took to execute the script.
    """
    def __init__(self, use_color=None, log_path=None):
        self.t0 = time.time()
        self.stamps = {}
        self.stamps['start'] = self.t0
        self.stamps['last'] = self.t0
        self.stat = {}
        self.verbose = 1
        if use_color is None:
            self.use_color = sys.stdout.isatty()
        else:
            self.use_color = use_color
        self.log_path = log_path
        if log_path is not None:
            self.log = open(log_path, 'w')

    def elapsed(self, name=None):
        if name is None:
            name = 'start'
        t = self.stamps[name]
        return time.time() - t
    
    def time(self):
        """
        Returns the elapsed time since the start or a specific name.
        """
        return self.elapsed(name='start')

    def start(self, message=None, name=None, verbose_lim=1):
        if name is None:
            name = 'last'
        self.stamps[name] = time.time()
        if message is not None:
            self.message(message, verbose_lim=verbose_lim)

    def message(self, message, verbose_lim=1):
        if verbose_lim <= self.verbose:
            time = self.elapsed()
            time_string = get_time_string(time, add_units=True)
            if self.use_color:
                print(f"{CYAN}[ {time_string} ]{RESET} {message}")
            else:
                print(f"[ {time_string} ] {message}")
            if self.log_path is not None:
                self.log.write(f"[ {time_string} ] {message}\n")

    def record(self, message=None, name=None, verbose_lim=1):
        if name is None:
            name = 'last'
        if verbose_lim <= self.verbose:
            time = self.elapsed()
            time_string = get_time_string(time, add_units=True)
            recorded_time = self.elapsed(name)
            recorded_time_string = get_time_string(recorded_time, add_units=True)
            if message is None:
                message = "Done."
            if self.use_color:
                print(f"{CYAN}[ {time_string} ]{RESET} {message} -> {GREEN}{recorded_time_string}{RESET}")
            else:
                print(f"[ {time_string} ] {message} -> {recorded_time_string}")
            if self.log_path is not None:
                self.log.write(f"[ {time_string} ] {message} -> {recorded_time_string}\n")
        if name not in self.stat:
            self.stat[name] = 0.0
        self.stat[name] += self.elapsed(name)

    def measure(self, func, message=None, *args, **kwargs):
        self.start(message)
        result = func(*args, **kwargs)
        self.record()
        return result

    def summary(self):
        print("Summary of time consumption:")
        for name, t in self.stat.items():
            time_string = get_time_string(t, add_units=True)
            print(f"  {CYAN}{name}{RESET}: {GREEN}{time_string}{RESET}")
        if self.log_path is not None:
            self.log.write("Summary of time consumption:\n")
            for name, t in self.stat.items():
                time_string = get_time_string(t, add_units=True)
                self.log.write(f"  {name}: {time_string}\n")
            self.log.close()


def get_time_string(elapsed_time, add_units=False, use_float=False):
    """
    Convert elapsed time in seconds to a formatted string.
    """
    time_format = "%H:%M:%S"
    if elapsed_time < 60:
        if add_units:
            return f"{elapsed_time:05.2f}s"
        else:
            return f"{elapsed_time:05.2f}"
    elif elapsed_time < 3600:
        if add_units:
            time_format = "%Mm %Ss"
        else:
            time_format = "%M:%S"
    elif elapsed_time < 86400:
        if add_units:
            time_format = "%Hh %Mm %Ss"
        else:
            time_format = "%H:%M:%S"
    else:
        elapsed_day = int(elapsed_time // 86400)  # Convert to days
        if add_units:
            time_format = f"{elapsed_day:2d}d %Hh %Mm %Ss"
        else:
            time_format = f"{elapsed_day:2d} %H:%M:%S"
    return time.strftime(time_format, time.gmtime(elapsed_time))
