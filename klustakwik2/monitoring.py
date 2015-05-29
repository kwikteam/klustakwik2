'''
Tools for monitoring KK as it runs

MonitoringServer and MonitoringClient are lightly adapted from Brian (briansimulator.org).
'''

try:
    import multiprocessing
    from multiprocessing.connection import Listener, Client
except ImportError:
    multiprocessing = None
import select
import inspect
from six import exec_

__all__ = ['MonitoringServer', 'MonitoringClient']


class MonitoringServer(object):
    '''
    Allows remote control (via IP) of a running KK script

    Initialisation arguments:

    ``server``
        The IP server details, a pair (host, port). If you want to allow control
        only on the one machine (for example, by an IPython shell), leave this
        as ``None`` (which defaults to host='localhost', port=2719). To allow
        remote control, use ('', portnumber).
    ``authkey``
        The authentication key to allow access, change it from 'klustakwik' if you
        are allowing access from outside (otherwise you allow others to run
        arbitrary code on your machine).
    ``global_ns``, ``local_ns``, ``level``
        Namespaces in which incoming commands will be executed or evaluated,
        if you leave them blank it will be the local and global namespace of
        the frame from which this function was called (if level=1, or from
        a higher level if you specify a different level here).

    Once this object has been created, use a :class:`MonitoringClient` to
    issue commands.

    **Example usage**

    Main script code includes a line like this::

        server = MonitoringServer()
        kk.register_callback(server)

    In an IPython shell you can do something like this::

        client = MonitoringClient()
        print client.evaluate('kk.num_clusters_alive')
    '''
    def __init__(self, server=None, authkey='klustakwik',
                 global_ns=None, local_ns=None, level=0):
        if multiprocessing is None:
            raise ImportError('Cannot import the required multiprocessing module.')
        if server is None:
            server = ('localhost', 2719)
        frame = inspect.stack()[level + 1][0]
        ns_global, ns_local = frame.f_globals, frame.f_locals
        if global_ns is None:
            global_ns = frame.f_globals
        if local_ns is None:
            local_ns = frame.f_locals
        self.local_ns = local_ns
        self.global_ns = global_ns
        self.listener = Listener(server, authkey=authkey)
        self.conn = None

    def __call__(self, kk):
        try:
            if self.conn is None:
                # This is kind of a hack. The multiprocessing.Listener class doesn't
                # allow you to tell if an incoming connection has been requested
                # without accepting that connection, which means if nothing attempts
                # to connect it will wait forever for something to connect. What
                # we do here is check if there is any incoming data on the
                # underlying IP socket used internally by multiprocessing.Listener.
                socket = self.listener._listener._socket
                sel, _, _ = select.select([socket], [], [], 0)
                if len(sel):
                    self.conn = self.listener.accept()
            if self.conn is None:
                return
            conn = self.conn
            global_ns = self.global_ns
            local_ns = self.local_ns
            paused = 1
            while conn and paused != 0:
                if paused >= 0 and not conn.poll():
                    return
                try:
                    job = conn.recv()
                except:
                    self.conn = None
                    break
                jobtype, jobargs = job
                if paused == 1: paused = 0
                try:
                    result = None
                    if jobtype == 'exec':
                        exec_(jobargs, global_ns, local_ns)
                    elif jobtype == 'eval':
                        result = eval(jobargs, global_ns, local_ns)
                    elif jobtype == 'setvar':
                        varname, varval = jobargs
                        local_ns[varname] = varval
                    elif jobtype == 'pause':
                        paused = -1
                    elif jobtype == 'go':
                        paused = 0
                except Exception as e:
                    # if it raised an exception, we return that exception and the
                    # client can then raise it.
                    result = e
                conn.send(result)
        except IOError:
            self.conn = None


class MonitoringClient(object):
    '''
    Used to remotely control (via IP) a running KlustaKwik script

    Initialisation arguments:

    ``server``
        The IP server details, a pair (host, port). If you want to allow control
        only on the one machine (for example, by an IPython shell), leave this
        as ``None`` (which defaults to host='localhost', port=2719). To allow
        remote control, use ('', portnumber).
    ``authkey``
        The authentication key to allow access, change it from 'klustakwik' if you
        are allowing access from outside (otherwise you allow others to run
        arbitrary code on your machine).

    Use a :class:`MonitoringServer` on the run you want to control.

    Has the following methods:

    .. method:: execute(code)

        Executes the specified code in the server process.
        If it raises an
        exception, the server process will catch it and reraise it in the
        client process.

    .. method:: evaluate(code)

        Evaluate the code in the server process and return the result.
        If it raises an
        exception, the server process will catch it and reraise it in the
        client process.

    .. method:: set(name, value)

        Sets the variable ``name`` (a string) to the given value (can be an
        array, etc.). Note that the variable is set in the local namespace, not
        the global one, and so this cannot be used to modify global namespace
        variables. To do that, set a local namespace variable and then
        call :meth:`~MonitoringClient.execute` with an instruction to change
        the global namespace variable.

    .. method:: pause()

        Temporarily stop the script in the server process, continue
        simulation with the :meth:'go' method.

    .. method:: go()

        Continue a script that was paused.

    .. method:: stop()

        Stop a script, equivalent to ``execute('exit(0)')``.

    **Example usage**

    Main script code includes a line like this::

        server = MonitoringServer()

    In an IPython shell you can do something like this::

        client = MonitoringClient()
        print client.evaluate('kk.num_clusters_alive')
   '''
    def __init__(self, server=None, authkey='klustakwik'):
        if multiprocessing is None:
            raise ImportError('Cannot import the required multiprocessing module.')
        if server is None:
            server = ('localhost', 2719)
        self.client = Client(server, authkey=authkey)

    def execute(self, code):
        self.client.send(('exec', code))
        result = self.client.recv()
        if isinstance(result, Exception):
            raise result

    def evaluate(self, code):
        self.client.send(('eval', code))
        result = self.client.recv()
        if isinstance(result, Exception):
            raise result
        return result

    def set(self, name, value):
        self.client.send(('setvar', (name, value)))
        result = self.client.recv()
        if isinstance(result, Exception):
            raise result

    def pause(self):
        self.client.send(('pause', ''))
        self.client.recv()

    def go(self):
        self.client.send(('go', ''))
        self.client.recv()

    def stop(self):
        self.execute('exit(0)')
