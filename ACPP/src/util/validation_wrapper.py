#!/usr/bin/python
'''
Parallel solver wrapper used for testing parallel/sequential solvers.
Invoke like:
python
testing_wrapper.py <testing_instance_file> <output file>
<cutoff time> <cutoff length> <seed> <param> ...
<param> : -name 'value'
'''
import os
import sys
import json
import re
from tempfile import NamedTemporaryFile
sys.path.append(os.path.abspath(os.path.realpath(__file__)+'/../../../'))
from execution import execution
from parameter_parser import parameter_parser


class testing(object):
    def __init__(self):
        self._thread_to_solver = dict()
        self._thread_to_params = dict()
        self._solver_log = dict()
        self._cmds = dict()
        self._sub_process = dict()
        self._verbose = False
        self._mem_limit = 12 * 1024 * 1024 * 1024  # 12GB
        self._log_path = sys.path[0] + "/valilogs/"
        self._watcher_path = self._log_path
        self.paramFile = dict()

    def parse_args(self, argv):
        ''' parse command line arguments
            Args:
                argv: command line arguments list

        '''
        self._instance = argv[1]
        self._output_file = argv[2]
        self._cutoff_time = float(argv[3])
        self._cutoff_length = int(argv[4])
        self._random_seed = argv[5]
        self._params = argv[6:]
        self._output = ""
        # Handle spaces in instance name
        # self._instance = self._instance.replace(" ", "\ ")

        # parse params to assign solvers and params to threads
        parser = parameter_parser()
        [self._thread_to_solver, self._thread_to_params, _] = parser.parse(
            self._params)

    def __construct_cmd(self, solver_path, parameter_list):
        # for lingeling-ala
        if "lingeling-ala" in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + "--seed=" + self._random_seed + " "
            cmd = cmd + self._instance + " "

            return cmd

        # for lingeling-ars
        if "lingeling-ars" in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + "--seed=" + self._random_seed + " "
            cmd = cmd + self._instance + " "

            return cmd

        # for clasp
        if 'clasp' in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + "--seed=" + self._random_seed + " "
            cmd = cmd + " -f " + self._instance + " "

            return cmd

        # for riss fixed seed
        if 'riss' in solver_path\
            or 'TNM' in solver_path\
                or 'MPhaseSAT_M' in solver_path\
                or 'march_hi' in solver_path\
                or 'bin/lingeling' in solver_path\
                or 'satUZK_wrapper' in solver_path\
                or 'glucose_wrapper' in solver_path\
                or 'contrasat' in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + self._instance + " "

            return cmd

        # for sparrow fixed seed: 0
        if 'sparrow' in solver_path:
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + self._instance + ' 0'

            return cmd

        # for GA-EAX
        if 'jikken' in solver_path:
            optimumFile = ('/home/liuwei/GAST/ACPP/instance_set/TSP/indices/'
                           'TSP_optimum.json')
            if 'AC_output/GAST' in self._instance:
                values = self._instance[self._instance.find(
                    'it')+2:].strip().split('/')
                if 'n' not in values[1]:
                    optimumFile = ('/home/liuwei/GAST/ACPP/AC_output/GAST/'
                                   'it%s/TSP_optimum.json') % values[0]
                else:
                    optimumFile = ('/home/liuwei/GAST/ACPP/AC_output/GAST/'
                                   'it%s/TSP_new_optimum.json') % values[0]
            with open(optimumFile, 'r') as f:
                optimum = json.load(f)
            cmd = solver_path + ' 10000 ' + ' 1.txt '
            for parameter in parameter_list:
                if 'populationsize' in parameter:
                    pSize = parameter.strip().split('=')[1]
                if 'offspringsize' in parameter:
                    oSize = parameter.strip().split('=')[1]
            cmd = cmd + ' ' + pSize + ' ' + oSize + ' '
            cmd = cmd + ' ' + optimum[self._instance] + ' '
            cmd = cmd + self._instance

            return cmd

        # for CLK-linkern
        if "linkern" in solver_path:
            optimumFile = ('/home/liuwei/GAST/ACPP/instance_set/TSP/indices/'
                           'TSP_optimum.json')
            if 'AC_output/GAST' in self._instance:
                values = self._instance[self._instance.find(
                    'it')+2:].strip().split('/')
                if 'n' not in values[1]:
                    optimumFile = ('/home/liuwei/GAST/ACPP/AC_output/GAST/'
                                   'it%s/TSP_optimum.json') % values[0]
                else:
                    optimumFile = ('/home/liuwei/GAST/ACPP/AC_output/GAST/'
                                   'it%s/TSP_new_optimum.json') % values[0]
            with open(optimumFile, 'r') as f:
                optimum = json.load(f)
            cmd = solver_path + " "
            for parameter in parameter_list:
                cmd = cmd + parameter + " "
            cmd = cmd + " -h " + optimum[self._instance]
            cmd = cmd + " -s 0 "
            cmd = cmd + self._instance + " "

            return cmd

        # for LKH
        if 'LKH' in solver_path:
            optimumFile = os.path.abspath(os.path.realpath(__file__)+'/../../../instance_set/TSP/indices/TSP_optimum.json')
            if 'AC_output/GAST' in self._instance:
                values = self._instance[self._instance.find(
                    'it')+2:].strip().split('/')
                if 'n' not in values[1]:
                    optimumFile = os.path.abspath((os.path.realpath(__file__)+'/../../../AC_output/GAST/it%s/TSP_optimum.json') % values[0])
                else:
                    optimumFile = os.path.abspath((os.path.realpath(__file__)+'/../../../AC_output/GAST/it%s/TSP_new_optimum.json') % values[0])
            if 'AC_output/PARHYDRA' in self._instance:
                values = self._instance[self._instance.find('it')+2:].strip().split('/')
                if 'n' not in values[1]:
                    optimumFile = os.path.abspath((os.path.realpath(__file__)+'/../../../AC_output/PARHYDRA/it%s/TSP_optimum.json') % values[0])
                else:
                    optimumFile = os.path.abspath((os.path.realpath(__file__)+'/../../../AC_output/PARHYDRA/it%s/TSP_new_optimum.json') % values[0])
            with open(optimumFile, 'r') as f:
                optimum = json.load(f)
            tmp = NamedTemporaryFile('w+b', prefix='Paramfile')
            self.paramFile[len(self.paramFile)] = tmp
            tmp.write('PROBLEM_FILE=%s\n' % self._instance)
            tmp.write('OPTIMUM=%s\n' % optimum[self._instance])
            for parameter in parameter_list:
                tmp.write(parameter[1:] + '\n')
            tmp.write('SEED=%s\n' % self._random_seed)
            tmp.flush()
            cmd = solver_path + ' ' + tmp.name

            return cmd

    def __read_output(self, runtime):
        output_line = []
        result = "TIMEOUT"
        for thread_id, log in self._solver_log.items():
            if os.path.isfile(log.name):
                if os.path.getsize(log.name) > 0:
                    log.seek(0, 0)
                    for line in log.readlines():
                        if line.startswith("s "):
                            if "UNSATISFIABLE" in line:
                                self._watcher_log.write(
                                    "\n" + str(thread_id) + ": solved\n")
                                result = "UNSAT"
                            elif "SATISFIABLE" in line:
                                self._watcher_log.write(
                                    str(thread_id) + ": solved\n")
                                result = "SAT"
                            elif "UNKNOWN" in line:
                                self._watcher_log.write(
                                    str(thread_id) + ": unsolved\n")
                            self._watcher_log.flush()

                            break
                        if line.startswith('Have hit the optimum') or\
                           line.startswith('Has hit the optimum'):
                            self._watcher_log.write(
                                    str(thread_id) + ": solved\n")
                            result = "SAT"
                            break
                        if line.startswith('Successes/Runs'):
                            rrr = re.search(r'\d+', line)
                            if rrr:
                                if int(rrr.group()) > 0:
                                    result = "SAT"
                                    break

                else:
                    self._watcher_log.write(
                        str(thread_id) + ": empty log unsolved\n")
                    self._watcher_log.flush()

                # delete this tempfile
                log.close()
            else:
                print log + "not exists"
                sys.exit()
        self._watcher_log.close()
        # Result for ParamILS: <solved>, <runtime>, <runlength>,
        # <quality>, <seed>, <additional rundata>
        output_line = "Result for SMAC: " + result + ", " + \
            str(runtime) + ", 0, 0, " + str(self._random_seed) + ", 0"

        return output_line

    def start(self):
        '''
        start solver
        '''
        # build watcher logs: temporary file
        self._watcher_log = NamedTemporaryFile(
            mode="w+b", prefix="watcher", dir=self._watcher_path)

        for thread_id in self._thread_to_solver.keys():
            # build log files: temporary file
            self._solver_log[thread_id] = NamedTemporaryFile(
                mode="w+b", prefix=("Thread" + str(thread_id) + "solverLog"),
                dir=self._log_path)

            self._cmds[thread_id] = self.__construct_cmd(
                self._thread_to_solver[thread_id],
                self._thread_to_params[thread_id]) + \
                " >& " + self._solver_log[thread_id].name

        # call execution wrapper
        exe = execution(self._cmds, self._cutoff_time,
                        self._mem_limit, self._watcher_log,
                        self._verbose)

        runtime = exe.run()
        with open(self._output_file, "w+") as f:
            f.write(self.__read_output(runtime))
            f.flush()


if __name__ == "__main__":
    Testing = testing()
    Testing.parse_args(sys.argv)
    Testing.start()
    if Testing.paramFile:
        for k, v in Testing.paramFile.items():
            v.close()
