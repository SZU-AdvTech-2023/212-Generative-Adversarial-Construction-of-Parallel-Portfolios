'''
Run cmds in paralle mode with time
and memory limits.
Once any subprocess ends, check if it
was exited normally or terminated

Input: cmds, cutoff time, mem limit
Output: excution result of each cmd
including: exit status
'''

import time
import os
from datetime import datetime
import re
import psutil


class execution(object):
    def __init__(self, cmds, cutoff_time,
                 mem_limit, watcher_log,
                 verbose=False, fast_solving_time=0.001):
        self._cmds = cmds  # dict
        self._cutoff_time = cutoff_time  # number in seconds
        self._mem_limit = 24 * 1024 * 1024 * 1024  # 24GB
        self._verbose = verbose
        self._sub_process = dict()
        self._watcher_log = watcher_log
        self._solver_log = dict()
        self._inq_freq = 0.1  # inqury frequency
        self._fast_solving_time = fast_solving_time
        self._sleeping = 0.01

    def run(self):
        for thread_id, cmd in self._cmds.items():
            logName = cmd[cmd.find('>&')+2:].strip()
            self._watcher_log.write("Thread %d thread log: %s\n"
                                    % (thread_id, logName))
            self._solver_log[thread_id] = logName
        self._watcher_log.flush()

        # Starting execution
        starting_time = time.time()
        for thread_id, cmd in self._cmds.items():
            if self._verbose:
                print "Now execute " + cmd
            self._sub_process[thread_id] = psutil.Popen(cmd, shell=True,
                                                        executable='/bin/bash')
        self._watcher_log.write(str(datetime.now()) + "\n")
        self._watcher_log.write("Parent process PID is " +
                                str(self._sub_process[1].ppid()) + "\n")
        self._watcher_log.write("Parent process PID is " +
                                str(os.getpid()) + " #using os.getpid() \n")
        self._watcher_log.write(
            "Sub process number :" + str(len(self._sub_process)) + "\n")
        self._watcher_log.flush()

        # Monitoring
        termination = False
        sucessStr = ['s UNSATISFIABLE', 's SATISFIABLE',
                     'Have hit the optimum', 'Has hit the optimum']
        while not termination:
            time.sleep(self._sleeping)
            elapsedTime = time.time() - starting_time
            if elapsedTime > self._cutoff_time:
                termination = True
                #print "time is out"
                break
            for thread_id, sub_process in self._sub_process.items():
                if sub_process.poll() is not None:
                    # check solver_log
                    self._watcher_log.write('Thread %d terminates\n' %
                                            thread_id)
                    self._watcher_log.flush()
                    with open(self._solver_log[thread_id], 'r') as f:
                        content = f.read()
                        if any(s in content for s in sucessStr):
                            # success
                            termination = True
                            self._watcher_log.write('Reason: success')
                            self._watcher_log.flush()
                            break
                        if content.find('Successes/Runs'):
                            line = content[content.find('Successes/Runs'):]
                            rrr = re.search(r'\d+', line)
                            if rrr:
                                if int(rrr.group()) > 0:
                                    termination = True
                                    self._watcher_log.write('Reason: success')
                                    self._watcher_log.flush()
                                    break
                        # crash
                        self._watcher_log.write('Reason: crash')
                        self._watcher_log.flush()
                        self._sub_process.pop(thread_id, 0)
            if not self._sub_process:
                #print('no thread running')
                self._watcher_log.write('No thread running\n')
                self._watcher_log.flush()
                termination = True
                break

        # termination and return
        self.__terminate()
        return min(self._cutoff_time, elapsedTime)

    def __terminate(self):
        # Terminates all threads
        self._watcher_log.write("Now we are terminating all threads\n")
        self._watcher_log.flush()

        for thread_id, sub_process in self._sub_process.items():
            # terminate child process
            self._watcher_log.write('Terminate solver process of thread %d' %
                                    thread_id)
            self._watcher_log.flush()
            try:
                children = sub_process.children(recursive=True)
            except psutil.NoSuchProcess:
                continue
            for p in children:
                try:
                    p.terminate()
                except(psutil.NoSuchProcess, KeyError):
                    # Key error means subprocess terminated before
                    # obtaining child process
                    pass
                else:
                    waiting_time = 0
                    time.sleep(self._sleeping)
                    waiting_time += self._sleeping

                    # ensure child process is terminated
                    while psutil.pid_exists(p.pid):
                        # Not terminated yet
                        self._watcher_log.write(
                            "Waiting " + str(waiting_time) +
                            " for solver process of thread " +
                            str(thread_id) + " terminated \n")
                        self._watcher_log.flush()
                        time.sleep(self._sleeping)
                        waiting_time += self._sleeping
                        if waiting_time >= 1 and waiting_time <= 2:
                            # SEND SIGTERM AGAIN
                            try:
                                p.terminate()
                            except psutil.NoSuchProcess:
                                pass
                        if waiting_time > 2:
                            # SEND SIGKILL
                            try:
                                p.kill()
                            except psutil.NoSuchProcess:
                                pass

            # terminate sub process
            self._watcher_log.write(
                "Trying to terminate subprocess of thread " +
                str(thread_id) + "\n")
            self._watcher_log.flush()

            try:
                sub_process.terminate()
            except psutil.NoSuchProcess:
                pass
            else:
                waiting_time = 0
                while sub_process.poll() is None:
                    # Not terminated yet
                    self._watcher_log.write(
                        "Waiting" + str(waiting_time) +
                        "for subprocess of thread " +
                        str(thread_id) + " terminated \n")
                    self._watcher_log.flush()
                    time.sleep(self._sleeping)
                    waiting_time += self._sleeping
                    if waiting_time >= 1 and waiting_time <= 2:
                        # SEND SIGTERM AGAIN
                        
                        try:
                            sub_process.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    if waiting_time > 2:
                        # SEND SIGKILL
                        try:
                            sub_process.kill()
                        except psutil.NoSuchProcess:
                            pass

            self._watcher_log.write("Thread " + str(thread_id) +
                                    " is terminated\n")
            self._watcher_log.flush()
