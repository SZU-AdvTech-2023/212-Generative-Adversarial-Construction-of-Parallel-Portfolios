'''
Configure component solver for P
'''
import subprocess
import os
import time
import random
import glob
import numpy as np
from datetime import datetime
# from validation import validation

project_dir = os.path.abspath(os.path.realpath(__file__)+'/../../')

# sys.path.append()
# Set parameter file and algorithm number
paramFile = project_dir+'/Solver/paramfile/Mul_lingeling_ala_pcs4.txt'

# paramFile = os.path.abspath(file_path+'/Solver/paramfile/Single_lkh_pcs.txt')
algNum = 4
# Set initial training instance index file
domain = 'SAT0'
mode = "small"
expNum = 1
instanceIndexFile = project_dir+('/instance_set/%s/indices/training') % domain
# instanceIndexFile = project_dir+('/instance_set/%s/indices/training_index_%s_%d') % (domain,mode,expNum)
featureFile= project_dir+('/instance_set/%s/indices/whole_instance_feature') % domain
# featureFile= None

# Set time options
configurationTime = 3600*30
validationTime = 3600*4
# Set target algorithm cutoff time
cutoffTime = 150

# Set random2seed generator helper
seedHelper = 42
# Set Algorithm configurator runs for each component solver
acRuns = 10

currentP = dict()
logFile = open("GLOBAL_log.txt", "w+")

initialInc = []

random.seed(seedHelper)
algNum = len(currentP)

logFile.write('Current we have %d Algs\nNeed %d AC runs\n' %
              (algNum, acRuns))
existingSolver = ''
for i in range(1, algNum+1):
    existingSolver = existingSolver + ' ' +\
                     currentP[i].replace('-@1', '-@%d' % i) + ' '
logFile.write('------Existing solver-------\n%s\n' % existingSolver)
logFile.flush()
logFile.write('--------------Training--------------------'
              '-------------------------------------------\n')

runs = range(1, acRuns+1)
# According to runs, execute AC to find component solver for P
# Refresh output folders
for runNum in runs:
    cmd1 = "rm -r "+project_dir+"/AC_output/GLOBAL/run" + \
        str(runNum) + "/output"
    cmd2 = "mkdir -p "+project_dir+"/AC_output/GLOBAL/run" + \
        str(runNum) + "/output"
    tmp = subprocess.Popen(cmd1, shell=True)
    tmp.communicate()
    tmp = subprocess.Popen(cmd2, shell=True)
    tmp.communicate()

# [con_scenario_file(runs, cutoffTime,
                  # instanceIndexFile, featureFile, paramFile)
# def con_scenario_file(runs, cutoffTime,
                      # instanceIndexFile, featureFile, paramFile):
training = instanceIndexFile
testing = training

for run_number in runs:
    scenarioFile = project_dir+('/AC_output/GLOBAL/run%d'
                    '/scenario.txt') % run_number
    FILE = open(scenarioFile, "w+")
    lines = []
    lines.append(('algo = '+project_dir+
                  '/AC/GAST_solver.py\n'))
    lines.append("execdir = /\n")
    lines.append("deterministic = 0\n")
    lines.append("run_obj = RUNTIME\n")
    lines.append("overall_obj = MEAN10\n")
    lines.append(("target_run_cputime_limit = " + str(cutoffTime) + "\n"))
    lines.append("paramfile = %s\n" % paramFile)
    lines.append(("instance_file = " + training + "\n"))
    if featureFile is not None:
        lines.append(("feature_file = " + featureFile + "\n"))
    lines.append(("test_instance_file = " + testing + "\n"))
    lines.append('outdir = '+project_dir+('/AC_output/GLOBAL'
                  '/run%d/output') %run_number)

    FILE.writelines(lines)
    FILE.close()
# ]


# obtain initialInc from fullconfigs
logFile.write('Executing %s runs\n' % str(runs))
# [run(runs, configurationTime, initialInc,
    # algNum, existingSolver, logFile)
# def run(runs, Timeout, initialInc, algNum, existingSolver,
        # logFile):
# Now we are at /smac

with open('exsiting_solver_Glo.txt', 'w+') as f:
    f.write(existingSolver + '\n')
    f.write('%d\n' % algNum)
# pool = set()
# os.chdir("/home/liuwei/GAST/ACPP/AC")
pool = set()
seedList = []
while len(seedList) <= len(runs):
    seed = random.randint(1, 10000000)
    if seed not in seedList:
        seedList.append(seed)
for i, run_number in list(enumerate(runs)):
    if initialInc:
        cmd = "./smac " + " --scenario-file " +\
              project_dir+('/AC_output/GLOBAL/run%d'
               '/scenario.txt') % run_number +\
              " --wallclock-limit " + \
              str(configurationTime) + " --seed " + str(seedList[i]) + \
              " --validation false " + \
              " --console-log-level OFF" + \
              " --log-level TRACE" + \
              " --initial-incumbent " + '"' + initialInc + ' "'
    else:
        cmd = "./smac " + " --scenario-file " +\
            project_dir+('/AC_output/GLOBAL/run%d'
             '/scenario.txt') % run_number +\
            " --wallclock-limit " + \
            str(configurationTime) + " --seed " + str(seedList[i]) + \
            " --validation false " + \
            " --console-log-level OFF" + \
            " --log-level TRACE"
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
# os.chdir('/home/liuwei/GAST/ACPP/src/GAST')
# ]

# After finishing All configuration runs, now gather data
# incumbent of each config run
# [configs, fullconfigs = gathering(acRuns)
# def gathering(acRuns):
    # fullConfigs are for initialize incumbent for SMAC
configs = dict()
fullConfigs = dict()
# fullConfigs = []
for run in range(1, acRuns + 1):
    outputDir = glob.glob(project_dir+
                           ("/AC_output/GLOBAL/run%d/output/run%d/log-run*.txt") %
                          (run, run))[0]
    with open(outputDir, "r") as FILE:
        lines = FILE.read().strip()
        lines = lines[lines.find('has finished'):]
        lines = lines[lines.find('-@1'):]
        configs[run] = lines.split('\n')[0]

    outputDir = glob.glob(project_dir+("/AC_output/GLOBAL/run%d/output/run%d/detailed-traj-run-*.csv") %
                          (run, run))[0]
    with open(outputDir, 'r') as f:
        line = f.read().strip().split('\n')[-1]
        line = line.replace(' ', '').replace('"', '')
        fullConfig = line.split(',')[5:-1]
        for j, value in list(enumerate(fullConfig)):
            fullConfig[j] = '-' + value.replace('=', ' ')
        fullConfigs[run] = ' '.join(fullConfig)

    # return configs, fullConfigs
# ]
logFile.write(str(fullConfigs))

# Then do validation
logFile.write('--------------Validation--------------------'
              '-------------------------------------------\n')
logFile.flush()

# [incIndex, initialIncIndex = validate(acRuns,
                                     # instanceIndexFile,
                                     # validationTime, cutoffTime,
                                     # algNum, existingSolver,
                                     # logFile)
# def validate(acRuns, instanceIndexFile,
             # validationTime, cutoffTime, algNum, existingSolver, logFile):

# validate each config, to determine the best one, incIndex
# and the one that improve the currentP at most, initialIncIndex
# [validation(instanceIndexFile, validationTime,
           # cutoffTime, algNum, acRuns, existingSolver,
           # logFile)
# def validation(insFile, budget, cutoffTime,
#                algNum, acRuns, existingSolver,
#                logFile):
insFile=instanceIndexFile
budget=validationTime
for i in range(1, acRuns+1):
    cmd = 'rm -r '+project_dir+('/validation_output/GLOBAL/run%d*' % i)
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
processes = set()
logFile.write('------Current we have %d Algs-------\n' % (algNum+1))

runs = range(1, (acRuns + 1))
logFile.write('Executing %s runs\n' % str(runs))
logFile.flush()
for runNum in runs:
    cmd = 'python GAST_validation.py %s %d %d %d %d %s' %\
          (insFile, budget, cutoffTime, runNum, algNum,\
           existingSolver)
    p = subprocess.Popen(cmd, shell=True)
    processes.add(p)

# waiting for validation finish
while processes:
    time.sleep(20)
    finished = [pid for pid in processes if pid.poll()
                is not None]
    processes -= set(finished)

with open(insFile, 'r') as f:
    instances = f.read().strip().split('\n')
insL = len(instances)

# compare validation results
outputdir = (project_dir+'/validation_output/GLOBAL/')
punish = 10
# performance matrix, [i,j] i+1 run j ins
perM = np.zeros((acRuns, insL)) * np.nan
runCount = np.zeros(perM.shape) * np.nan
# write to /validation_output/GAST/validation_results.txt
fileList = os.listdir(outputdir)
for f in fileList:
    if 'run' in f:
        begin = f.find('n')
        end = f.find('_')
        run_number = int(f[begin + 1: end])
        begin = f.find('s')
        end = f.find('S') - 1
        ins_index = int(f[begin + 1: end])
        with open(outputdir + f, 'r') as f:
            outPut = f.read().strip()
            values = outPut[outPut.find(
                ':') + 1:].strip().replace(' ', '').split(',')
        (result, runtime) = (values[0], float(values[1]))
        if 'TIMEOUT' in result:
            runtime = runtime * punish
        if np.isnan(perM[run_number-1, ins_index]):
            perM[run_number-1, ins_index] = runtime
            runCount[run_number-1, ins_index] = 1
        else:
            perM[run_number-1, ins_index] += runtime
            runCount[run_number-1, ins_index] += 1
final_results = []
perM = np.true_divide(perM, runCount)
for row in perM:
    tagshape = ~np.isnan(row)
    final_results.append(np.mean(row[tagshape]))
logFile.write('Validation results\n%s\n' % str(final_results))
logFile.flush()
# incIndex
incIndex = np.argmin(final_results) + 1
# initialIncIndex
incRow = perM[incIndex-1, :]
np.save((project_dir+'/validation_output/GLOBAL/performance_matrix.npy'), incRow)
incRow = incRow[~np.isnan(incRow)]
bestValue = 1000000
initialIncIndex = -1
for i, row in list(enumerate(perM)):
    if i == incIndex - 1:
        continue
    row = row[~np.isnan(row)]
    if len(incRow) >= len(row):
        tmp = np.hstack((row, np.zeros(len(incRow)-len(row))*10000))
        tmpvalue = np.mean(np.minimum(incRow, tmp))
    else:
        tmp = np.hstack((incRow, np.zeros(len(row)-len(incRow))*10000))
        tmpvalue = np.mean(np.minimum(tmp, row))
    if tmpvalue < bestValue:
        bestValue = tmpvalue
        initialIncIndex = i + 1
outputdir = glob.glob(project_dir+"/AC_output/GLOBAL/run%d/output/run%d/log-run*.txt" %
                      (incIndex, incIndex))[0]
with open(outputdir, "r") as FILE:
    lines = FILE.read().strip()
    lines = lines[lines.find('has finished'):]
    lines = lines[lines.find('-@1'):]
    solver = lines.split('\n')[0]

result_file = (project_dir+'/validation_output/GLOBAL/validation_results.txt')
with open(result_file, 'w+') as f:
    f.write(str(incIndex) + '\n')
    f.write(str(initialIncIndex) + '\n')
    f.write(str(final_results) + '\n')
    f.write(str(perM) + '\n')
    f.write(str(runCount) + '\n')
    f.write(solver)
logFile.write('Incindex, initialIncindex are \n')
logFile.write(str(incIndex) + '\n')
logFile.write(str(initialIncIndex) + '\n')
# ]

# organize validation results
targetF = project_dir+'/validation_output/GLOBAL/it%s' % (algNum+1)
cmd = 'mkdir %s' % targetF
p = subprocess.Popen(cmd, shell=True)
p.communicate()

cmd = 'mv '+project_dir+'/validation_output/GLOBAL/run* %s/' % targetF
p = subprocess.Popen(cmd, shell=True)
p.communicate()
cmd = 'mv '+project_dir+('/validation_output/'
       'GLOBAL/validation_results.txt %s/') % targetF
p = subprocess.Popen(cmd, shell=True)
p.communicate()
cmd = 'mv '+project_dir+('/validation_output/'
       'GLOBAL/performance_matrix.npy %s/') % targetF
p = subprocess.Popen(cmd, shell=True)
p.communicate()

# read validation results
# obtain incIndex and initialIncIndex
fName = '%s/validation_results.txt' % targetF
with open(fName, 'r') as f:
    lines = f.readlines()
    incIndex = int(lines[0].strip())
    initialIncIndex = int(lines[1].strip())
    # return incIndex, initialIncIndex
# ]
currentP[algNum+1] = configs[incIndex]
# return currentP, fullconfigs[initialIncIndex]
