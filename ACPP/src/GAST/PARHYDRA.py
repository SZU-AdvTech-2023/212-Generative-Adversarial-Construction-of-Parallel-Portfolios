'''
Main components:
Parhydra + Instance Generation
'''
import subprocess
import os
import sys
import json
import time
import random
import glob
import psutil
import numpy as np
from datetime import datetime
from validation import validation
from comSearch import comSearch
#from instanceGeneration import insGen

#realpath,dirname,
file_path=os.path.realpath(__file__)
file_path=os.path.dirname(file_path)
file_path=os.path.abspath(file_path+"/../..")
#os.path.abspath(os.path.realpath(__file__)+"/../../..")#ACPP path

# Set parameter file and algorithm number
paramFile = file_path+'/Solver/paramfile/Single_lingeling_ala_pcs.txt'
# paramFile = os.path.abspath(file_path+'/Solver/paramfile/Single_lkh_pcs.txt')
algNum = 4
# Set initial training instance index file
domain = 'SAT0'
mode = "small"
expNum = 1
configurationTime=0
validationTime=0
cutoffTime=0

instanceIndexFile_Original = os.path.abspath(file_path + '/instance_set/%s/indices/training') % domain
featureFile = os.path.abspath(file_path+'/instance_set/%s/indices/whole_instance_feature')%domain
if domain == 'TSP':
    featureFile = None
    configurationTime = 5400#3600*1.5
    validationTime = 1800#3600*0.5
    #generationTime = 18000#3600*5
    cutoffTime=1 #TSP


if domain == 'SAT0':
# Set time options for SAT
    configurationTime = 3600*8
    validationTime = 3600*4
    # generationTime = 3600*4
    cutoffTime = 150 #SAT

# Set time options for test
#configurationTime = 360
#validationTime = 360
#generationTime = 360

# Set random2seed generator helper
seedHelper = 42
# Set Algorithm configurator runs for each component solver
acRuns = 10
# Set instance generation options
minTestTimes = 5
maxIt = 4
# parent selection, only for TSP, tournament or equal
# for SAT, we always use uniform
pSelection = 'uniform'
# survivior selection, for TSP and SAT
# tournament or truncation
surSelection = 'truncation'
# mutation rate, only for TSP
mu = 0.2
# changeP: delete a city, add a city, otherwise pure mutate
changeP = [0.1, 0.3]
cityLowerB, cityUpperB = [400, 600]
# whether reteining old instances
retein = True
# new instance size, only useful whtn retein is True
# in this case, select |newSize| instances from all
# instances and addup all original instances
newSize = 32


currentP = dict()
logFile = open("PARHYDRA_log.txt", "w+")
initialInc = []

cmd = 'rm -r '+os.path.abspath(file_path+'/validation_output/PARHYDRA/it*')
p = subprocess.Popen(cmd, shell=True)
p.communicate()
cmd = 'rm -r '+os.path.abspath(file_path+'/AC_output/PARHYDRA/it*')
p = subprocess.Popen(cmd, shell=True)
p.communicate()

for runNum in range(1, algNum+1):
    cmd = 'mkdir -p '+os.path.abspath(file_path+'/AC_output/PARHYDRA/it%d' % runNum)
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

for runNum_iter in range(1, algNum+1):
    logFile.write('-------------------------------------------\n')

    logFile.write('---------------Iteration %d ---------------\n' % runNum_iter)
    instanceIndexFile = (file_path+'/AC_output/PARHYDRA/it%d/training_instances') % runNum_iter
    if runNum_iter == 1:
        # instanceIndexFile = file_path+'/AC_output/PARHYDRA/it%d/training_instances' % (algNumc+1)
    # with open(instanceIndexFile, 'w+') as f:
        # for ins in instances:
            # f.write('%s\n' % ins)
        cmd = ('cp %s %s')% (instanceIndexFile_Original,instanceIndexFile)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

        
        with open(instanceIndexFile, 'r') as f:
            instances = f.read().strip().replace('\r', '').split('\n')
        newInstances = []
        for i, ins in enumerate(instances):
            cmd = 'cp %s '%ins+(file_path+'/AC_output/PARHYDRA/it1/%d' % (i+1))
            subprocess.check_output(cmd, shell=True)
            newInstances.append('\"'+os.path.abspath(file_path+'/AC_output/PARHYDRA/it1/%d'% (i+1))+'\"')
        with open(instanceIndexFile, 'w+') as f:
            for ins in newInstances:
                f.write(ins + '\n')

        if 'lkh' in paramFile:
            with open(os.path.abspath(file_path+'/instance_set/TSP/indices/TSP_optimum.json'), 'r') as f:
                optimum = json.load(f)
            newOptimum = dict()
            for i, ins in enumerate(instances):
                newOptimum[newInstances[i].replace('"', '')] = optimum[ins]
            with open(os.path.abspath(file_path+'/AC_output/PARHYDRA/it1/TSP_optimum.json'), 'w+') as f:
                json.dump(newOptimum, f)
        elif 'lingeling' in paramFile:
            featureLineDict = dict()
            with open(featureFile, 'r') as f:
                featureLines = f.read().strip().split('\n')
                featureLineDict['firstLine'] = featureLines[0]
                for line in featureLines[1:]:
                    splitValues = line.split(',')
                    insName = splitValues[0].replace('"', '')
                    featureLineDict[insName] = ','.join(splitValues[1:])
            featureFile = os.path.abspath(file_path+'/AC_output/PARHYDRA/it1/instance_feature')
            with open(featureFile, 'w+') as f:
                f.write(featureLineDict['firstLine'] + '\n')
                for i, ins in enumerate(instances):
                    insName = '\"'+os.path.abspath(file_path+'/AC_output/PARHYDRA/it1/%d' % (i+1))+'\"'
                    f.write(insName + ',' + featureLineDict[ins] + '\n')
    else:# 
        #copy from former iteration
        if 'lkh' in paramFile:
            cmd = "cp "+file_path+'/AC_output/PARHYDRA/it'+str(runNum_iter-1)+'/TSP_optimum.json'+' '+file_path+'/AC_output/PARHYDRA/it'+str(runNum_iter)+'/TSP_optimum.json'
            tmp = subprocess.Popen(cmd, shell=True)
            tmp.communicate()
        elif 'lingeling' in paramFile:
            cmd = "cp "+file_path+'/AC_output/PARHYDRA/it'+str(runNum_iter-1)+'/instance_feature'+' '+file_path+'/AC_output/PARHYDRA/it'+str(runNum_iter)+'/instance_feature'
            tmp = subprocess.Popen(cmd, shell=True)
            tmp.communicate()
        cmd = "cp "+file_path+'/AC_output/PARHYDRA/it'+str(runNum_iter-1)+'/training_instances'+' '+file_path+'/AC_output/PARHYDRA/it'+str(runNum_iter)+'/training_instances'
        tmp = subprocess.Popen(cmd, shell=True)
        tmp.communicate()
    #print("finished")
    #####################################################comSearch
    #currentP, initialInc = comSearch(currentP, initialInc, configurationTime,
    #                                 validationTime,
    #                                 cutoffTime, acRuns, instanceIndexFile,
    #                                 paramFile, featureFile, logFile,
    #                                 seedHelper)
    random.seed(seedHelper)
    algNumc = len(currentP)
    logFile.write('Current we have %d Algs\nNeed %d AC runs\n' % (algNumc, acRuns))
    existingSolver = ''
    for i in range(1, algNumc+1):
        existingSolver = existingSolver + ' ' +\
                         currentP[i].replace('-@1', '-@%d' % i) + ' '
    logFile.write('------Existing solver-------\n%s\n' % existingSolver)
    logFile.flush()
    logFile.write('--------------Training---------------------------------------\n')
    runs = range(1, acRuns+1)
    # According to runs, execute AC to find component solver for P
    # Refresh output folders
    for runNum in runs:
        cmd1 = "rm -r "+file_path+"/AC_output/PARHYDRA/run" + \
            str(runNum) + "/output"
        cmd2 = "mkdir -p "+file_path+"/AC_output/PARHYDRA/run" + \
            str(runNum) + "/output"
        tmp = subprocess.Popen(cmd1, shell=True)
        tmp.communicate()
        tmp = subprocess.Popen(cmd2, shell=True)
        tmp.communicate()

    ######################construct the scenario file for smac
    #con_scenario_file(runs, cutoffTime, instanceIndexFile, featureFile, paramFile)
    training = instanceIndexFile
    testing = training
    for run_number in runs:
        scenarioFile = file_path+('/AC_output/PARHYDRA/run%d'
                        '/scenario.txt') % run_number
        FILE = open(scenarioFile, "w+")
        lines = []
        lines.append(('algo = '+file_path+'/src/GAST/GAST_solver.py'+'\n'))
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
        lines.append('outdir = '+file_path+('/AC_output/PARHYDRA/run%d/output') %run_number)

        FILE.writelines(lines)
        FILE.close()
    ######################con_scenario_file
    ###
    # obtain initialInc from fullconfigs
    logFile.write('Executing %s runs\n' % str(runs))

    ############################################
    ######################run configuration
    # run(runs, configurationTime, initialInc,
       # algNum, existingSolver, logFile)
    ######################run configuration
    ############################################

    Timeout=configurationTime
    # Now we are at /smac
    with open('exsiting_solver_Par.txt', 'w+') as f:
        f.write(existingSolver + '\n')
        f.write('%d\n' % algNumc)
    pool = set()
    os.chdir(file_path+"/AC")
    pool = set()
    seedList = []
    while len(seedList) <= len(runs):
        seed = random.randint(1, 10000000)
        if seed not in seedList:
            seedList.append(seed)
    for i, run_number in list(enumerate(runs)):
        if initialInc:
            cmd = "./smac " + " --scenario-file " +\
                  file_path+('/AC_output/PARHYDRA/run%d'
                   '/scenario.txt') % run_number +\
                  " --wallclock-limit " + \
                  str(Timeout) + " --seed " + str(seedList[i]) + \
                  " --validation false " + \
                  " --console-log-level OFF" + \
                  " --log-level TRACE" + \
                  " --initial-incumbent " + '"' + initialInc + ' "'
        else:
            cmd = "./smac " + " --scenario-file " +\
                file_path+('/AC_output/PARHYDRA/run%d/scenario.txt') % run_number +\
                " --wallclock-limit " + \
                str(Timeout) + " --seed " + str(seedList[i]) + \
                " --validation false " + \
                " --console-log-level OFF" + \
                " --log-level TRACE"
        # exit()
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
    os.chdir(file_path+'/src/GAST')#here in the AC directory
    ######################run
    

    #####################################begin gathering
    # After finishing All configuration runs, now gather data
    # incumbent of each config run
    #configs, fullconfigs = gathering(acRuns)
    # fullConfigs are for initialize incumbent for SMAC
    configs = dict()
    fullconfigs = dict()
    for run in range(1, acRuns + 1):
        outputDir = glob.glob(file_path+("/AC_output/PARHYDRA/run%d/output/run%d/log-run*.txt") %
                              (run, run))[0]
        with open(outputDir, "r") as FILE:
            lines = FILE.read().strip()
            lines = lines[lines.find('has finished'):]
            lines = lines[lines.find('-@1'):]
            configs[run] = lines.split('\n')[0]

        outputDir = glob.glob(file_path+("/AC_output/PARHYDRA/run%d/output/run%d/detailed-traj-run-*.csv") %
                              (run, run))[0]
        with open(outputDir, 'r') as f:
            line = f.read().strip().split('\n')[-1]
            line = line.replace(' ', '').replace('"', '')
            fullConfig = line.split(',')[5:-1]
            for j, value in list(enumerate(fullConfig)):
                fullConfig[j] = '-' + value.replace('=', ' ')
            fullconfigs[run] = ' '.join(fullConfig)

    #return configs, fullconfigs
    #####################################end gathering
    
    
    #exit()

    # Then do validation
    logFile.write('--------------Validation--------------------'
                  '-------------------------------------------\n')
    logFile.flush()
    ####################################begin validate
    #incIndex, initialIncIndex = validate(acRuns,
    #                                     instanceIndexFile,
    #                                     validationTime, cutoffTime,
    #                                     algNum, existingSolver,
    #                                     logFile)
    # validate each config, to determine the best one, incIndex
    # and the one that improve the currentP at most, initialIncIndex
    ############begin validation
    #validation(instanceIndexFile, validationTime,
    #           cutoffTime, algNum, acRuns, existingSolver,
    #           logFile)
    insFile=instanceIndexFile
    budget=validationTime
    for i in range(1, acRuns+1):
        cmd = ('rm -r '+file_path+'/validation_output/PARHYDRA/run%d*' % i)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
    processes = set()
    logFile.write('------Current we have %d Algs-------\n' % (algNumc+1))

    runs = range(1, (acRuns + 1))
    logFile.write('Executing %s runs\n' % str(runs))
    logFile.flush()
    for runNum in runs:
        cmd = 'python PARHYDRA_validation.py %s %d %d %d %d %s' %\
              (insFile, budget, cutoffTime, runNum, algNumc,\
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
    outputdir = file_path+'/validation_output/PARHYDRA/'
    punish = 10
    # performance matrix, [i,j] i+1 run j ins
    perM = np.zeros((acRuns, insL)) * np.nan
    runCount = np.zeros(perM.shape) * np.nan
    # write to /validation_output/PARHYDRA/validation_results.txt
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
    np.save(file_path+('/validation_output/PARHYDRA/performance_matrix.npy'), incRow)
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
    outputdir = glob.glob(file_path+"/AC_output/PARHYDRA/run%d/output/run%d/log-run*.txt" %
                          (incIndex, incIndex))[0]
    with open(outputdir, "r") as FILE:
        lines = FILE.read().strip()
        lines = lines[lines.find('has finished'):]
        lines = lines[lines.find('-@1'):]
        solver = lines.split('\n')[0]

    result_file = file_path+'/validation_output/PARHYDRA/validation_results.txt'
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
    ############end validation
    

    # organize validation results
    targetF = file_path+'/validation_output/PARHYDRA/it%s' % (algNumc+1)
    cmd = 'mkdir %s' % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

    cmd = 'mv '+file_path+'/validation_output/PARHYDRA/run*'+' %s/' % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    cmd = 'mv '+file_path+'/validation_output/PARHYDRA/validation_results.txt %s/' % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    cmd = 'mv '+file_path+'/validation_output/PARHYDRA/performance_matrix.npy %s/' % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

    # read validation results
    # obtain incIndex and initialIncIndex
    fName = '%s/validation_results.txt' % targetF
    with open(fName, 'r') as f:
        lines = f.readlines()
        incIndex = int(lines[0].strip())
        initialIncIndex = int(lines[1].strip())
    #return incIndex, initialIncIndex
    ####################################end validate

    currentP[algNumc+1] = configs[incIndex]
    #return currentP, 
    initialInc=fullconfigs[initialIncIndex]
    #####################################################comSearch

    
    logFile.write('End iteration Current solver is \n%s\n' % str(currentP))
    logFile.flush()


fullSolver = []
for runNum in range(1, algNum + 1):
    fullSolver.append(currentP[runNum].replace('-@1', '-@%d' % (runNum)))
fullSolver = ' '.join(fullSolver)
logFile.write('Final solver:\n%s' % fullSolver)
logFile.close()
