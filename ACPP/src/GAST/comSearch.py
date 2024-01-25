'''
Configure component solver for P
'''
import subprocess
import os
import time
import random
import glob
from datetime import datetime
from validation import validation


def con_scenario_file(runs, cutoffTime,
                      instanceIndexFile, featureFile, paramFile):
    training = instanceIndexFile
    testing = training

    for run_number in runs:
        scenarioFile = ('/home/liuwei/GAST/ACPP/AC_output/GAST/run%d'
                        '/scenario.txt') % run_number
        FILE = open(scenarioFile, "w+")
        lines = []
        lines.append(('algo = /home/liuwei/GAST/ACPP/src/GAST/'
                      'GAST_solver.py\n'))
        lines.append("execdir = ./\n")
        lines.append("deterministic = 0\n")
        lines.append("run_obj = RUNTIME\n")
        lines.append("overall_obj = MEAN10\n")
        lines.append(("target_run_cputime_limit = " + str(cutoffTime) + "\n"))
        lines.append("paramfile = %s\n" % paramFile)
        lines.append(("instance_file = " + training + "\n"))
        if featureFile is not None:
            lines.append(("feature_file = " + featureFile + "\n"))
        lines.append(("test_instance_file = " + testing + "\n"))
        lines.append(('outdir = /home/liuwei/GAST/ACPP/AC_output/GAST'
                      '/run%d/output') % run_number)

        FILE.writelines(lines)
        FILE.close()


def run(runs, Timeout, initialInc, algNum, existingSolver,
        logFile):
    # Now we are at /smac
    with open('exsiting_solver.txt', 'w+') as f:
        f.write(existingSolver + '\n')
        f.write('%d\n' % algNum)
    # pool = set()
    os.chdir(os.path.abspath('../../AC'))
    # os.chdir("/home/liuwei/GAST/ACPP/AC")
    pool = set()
    seedList = []
    while len(seedList) <= len(runs):
        seed = random.randint(1, 10000000)
        if seed not in seedList:
            seedList.append(seed)
    for i, run_number in list(enumerate(runs)):
        if initialInc:
            cmd = "./smac " + " --scenario-file " + \
                  os.path.abspath('../../') + ('/ACPP/AC_output/GAST/run%d'
                                               '/scenario.txt') % run_number + \
                  " --wallclock-limit " + \
                  str(Timeout) + " --seed " + str(seedList[i]) + \
                  " --validation false " + \
                  " --console-log-level OFF" + \
                  " --log-level TRACE" + \
                  " --initial-incumbent " + '"' + initialInc + ' "'
        else:
            cmd = "./smac " + " --scenario-file " + \
                  os.path.abspath('../../') + ('/ACPP/AC_output/GAST/run%d'
                                               '/scenario.txt') % run_number + \
                  " --wallclock-limit " + \
                  str(Timeout) + " --seed " + str(seedList[i]) + \
                  " --validation false " + \
                  " --console-log-level OFF" + \
                  " --log-level TRACE"
        # exit(0)

        pool.add(subprocess.Popen(cmd, shell=True))

    finished = False
    estimated_time = 0
    while not finished:
        time.sleep(20)
        estimated_time += 20
        Finished_pid = [pid for pid in pool if pid.poll() is not None]
        pool -= set(Finished_pid)
        if not bool(pool):
            finished = True
        if estimated_time % 600 == 0:
            logFile.write(str(datetime.now()) + "\n")
            logFile.write("Now " + str(len(pool)) + " AC" + " are running\n")
            logFile.flush()
            cmd = 'free -m'
            logFile.write(str(subprocess.check_output(cmd, shell=True)))
            logFile.flush()
            logFile.write("Now running tasks: " +
                          subprocess.check_output("ps r|wc -l", shell=True))
            logFile.flush()
    os.chdir(os.path.dirname(__file__))  ###
    # os.chdir(os.path.realpath(__file__))


def gathering(acRuns):
    # fullConfigs are for initialize incumbent for SMAC
    configs = dict()
    fullConfigs = dict()
    for run in range(1, acRuns + 1):
        outputDir = glob.glob(("/home/liuwei/GAST/ACPP/AC_output/"
                               "GAST/run%d/output/run%d/log-run*.txt") %
                              (run, run))[0]
        with open(outputDir, "r") as FILE:
            lines = FILE.read().strip()
            lines = lines[lines.find('has finished'):]
            lines = lines[lines.find('-@1'):]
            configs[run] = lines.split('\n')[0]

        outputDir = glob.glob(("/home/liuwei/GAST/ACPP/AC_output/"
                               "GAST/run%d/output/run%d/detailed-traj-run-*.csv") %
                              (run, run))[0]
        with open(outputDir, 'r') as f:
            line = f.read().strip().split('\n')[-1]
            line = line.replace(' ', '').replace('"', '')
            fullConfig = line.split(',')[5:-1]
            for j, value in list(enumerate(fullConfig)):
                fullConfig[j] = '-' + value.replace('=', ' ')
            fullConfigs[run] = ' '.join(fullConfig)

    return configs, fullConfigs


def validate(acRuns, instanceIndexFile,
             validationTime, cutoffTime, algNum, existingSolver, logFile):
    # validate each config, to determine the best one, incIndex
    # and the one that improve the currentP at most, initialIncIndex
    validation(instanceIndexFile, validationTime,
               cutoffTime, algNum, acRuns, existingSolver,
               logFile)
    # organize validation results

    targetF = '/home/liuwei/GAST/ACPP/validation_output/GAST/it%s' % (algNum + 1)
    cmd = 'mkdir %s' % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

    cmd = 'mv /home/liuwei/GAST/ACPP/validation_output/GAST/run* %s/' % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    cmd = ('mv /home/liuwei/GAST/ACPP/validation_output/'
           'GAST/validation_results.txt %s/') % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    cmd = ('mv /home/liuwei/GAST/ACPP/validation_output/'
           'GAST/performance_matrix.npy %s/') % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

    # read validation results
    # obtain incIndex and initialIncIndex
    fName = '%s/validation_results.txt' % targetF
    with open(fName, 'r') as f:
        lines = f.readlines()
        incIndex = int(lines[0].strip())
        initialIncIndex = int(lines[1].strip())
    return incIndex, initialIncIndex


def comSearch(currentP, initialInc, configurationTime,
              validationTime,
              cutoffTime, acRuns, instanceIndexFile,
              paramFile, featureFile, logFile,
              seedHelper):
    random.seed(seedHelper)
    algNum = len(currentP)

    logFile.write('Current we have %d Algs\nNeed %d AC runs\n' %
                  (algNum, acRuns))
    existingSolver = ''
    for i in range(1, algNum + 1):
        existingSolver = existingSolver + ' ' + \
                         currentP[i].replace('-@1', '-@%d' % i) + ' '
    logFile.write('------Existing solver-------\n%s\n' % existingSolver)
    logFile.flush()
    logFile.write('--------------Training--------------------'
                  '-------------------------------------------\n')

    runs = range(1, acRuns + 1)
    # According to runs, execute AC to find component solver for P
    # Refresh output folders
    for runNum in runs:
        cmd1 = "rm -r /home/liuwei/GAST/ACPP/AC_output/GAST/run" + \
               str(runNum) + "/output"
        cmd2 = "mkdir /home/liuwei/GAST/ACPP/AC_output/GAST/run" + \
               str(runNum) + "/output"
        tmp = subprocess.Popen(cmd1, shell=True)
        tmp.communicate()
        tmp = subprocess.Popen(cmd2, shell=True)
        tmp.communicate()

    con_scenario_file(runs, cutoffTime,
                      instanceIndexFile, featureFile, paramFile)
    # print 10086
    # exit(0)
    # obtain initialInc from fullconfigs
    logFile.write('Executing %s runs\n' % str(runs))
    # print 10086
    # exit(0)
    run(runs, configurationTime, initialInc,
        algNum, existingSolver, logFile)

    # After finishing All configuration runs, now gather data
    # incumbent of each config run
    configs, fullconfigs = gathering(acRuns)

    # Then do validation
    logFile.write('--------------Validation--------------------'
                  '-------------------------------------------\n')
    logFile.flush()

    incIndex, initialIncIndex = validate(acRuns,
                                         instanceIndexFile,
                                         validationTime, cutoffTime,
                                         algNum, existingSolver,
                                         logFile)
    currentP[algNum + 1] = configs[incIndex]

    return currentP, fullconfigs[initialIncIndex]
