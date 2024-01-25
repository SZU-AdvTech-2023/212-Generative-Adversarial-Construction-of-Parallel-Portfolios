'''
Main components:
Parhydra + Instance Generation
'''
import os.path
import sys
import subprocess
import json
from comSearch import comSearch
from instanceGeneration import insGen

# Set parameter file and algorithm number
paramFile = '/home/liuwei/GAST/ACPP/Solver/paramfile/Single_lingeling_ala_pcs.txt'
# paramFile = '/home/liuwei/GAST/ACPP/Solver/paramfile/Single_lkh_pcs.txt'
# algNum = 2
algNum = 4
# Set initial training instance index file
domain = 'SAT0'
mode = "small"
expNum = 1
# instanceIndexFile = '/home/liuwei/GAST/ACPP/instance_set/%s/indices/training_index_%s_%d' %\
#                     (domain, mode, expNum)
instanceIndexFile = '/home/liuwei/GAST/ACPP/instance_set/%s/indices/training' %(domain)

featureFile = '/home/liuwei/GAST/ACPP/instance_set/%s/indices/whole_instance_feature' %(domain)
if domain == 'TSP':
    featureFile = None

# Set time options
configurationTime = 30
# configurationTime = 3600*8
validationTime = 30
# validationTime = 3600*4
generationTime = 30
# generationTime = 3600*40
# Set target algorithm cutoff time
cutoffTime = 150
# cutoffTime = 1
# Set random2seed generator helper
# seedHelper = 21
seedHelper = 42
# Set Algorithm configurator runs for each component solver
# acRuns = 4
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
# new instance size, only useful when retein is True
# in this case, select |newSize| instances from all
# instances and addup all original instances
newSize = 32


currentP = dict()
logFile = open("GAST_log.txt", "w+")
initialInc = []

cmd = 'rm -r /home/liuwei/GAST/ACPP/validation_output/GAST/it*'
p = subprocess.Popen(cmd, shell=True)
p.communicate()
cmd = 'rm -r /home/liuwei/GAST/ACPP/AC_output/GAST/it*'
p = subprocess.Popen(cmd, shell=True)
p.communicate()

for runNum in range(1, algNum+1):
    cmd = 'mkdir /home/liuwei/GAST/ACPP/AC_output/GAST/it%d' % runNum
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

for runNum in range(1, algNum+1):
    logFile.write('-------------------------------------------\n')
    logFile.write('---------------Iteration %d ---------------\n' % runNum)
    if runNum == 1:
        cmd = ('cp %s /home/liuwei/GAST/ACPP/AC_output/'
               'GAST/it1/training_instances') % instanceIndexFile
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

        instanceIndexFile = ('/home/liuwei/GAST/ACPP/AC_output/'
                             'GAST/it1/training_instances')
        with open(instanceIndexFile, 'r') as f:
            instances = f.read().strip().replace('\r', '').split('\n')

        newInstances = []

        for i, ins in enumerate(instances):
            # tempSign=os.path.basename(ins)
            # cmd = ('cp %s /home/liuwei/GAST/ACPP/AC_output/'
            #        'GAST/it1/%s' % (ins, tempSign))
            # subprocess.check_output(cmd, shell=True)
            # newInstances.append('\"/home/liuwei/GAST/ACPP/AC_output/'
            #                     'GAST/it1/%s\"' % (tempSign))
            cmd = ('cp %s /home/liuwei/GAST/ACPP/AC_output/'
                   'GAST/it1/%d' % (ins, i+1))
            subprocess.check_output(cmd, shell=True)
            newInstances.append('\"/home/liuwei/GAST/ACPP/AC_output/'
                                'GAST/it1/%d\"' % (i+1))

        with open(instanceIndexFile, 'w+') as f:
            for ins in newInstances:
                f.write(ins + '\n')

        if 'lkh' in paramFile:
            with open('/home/liuwei/GAST/ACPP/instance_set/TSP/indices/'
                      'TSP_optimum.json', 'r') as f:
                optimum = json.load(f)
            newOptimum = dict()
            for i, ins in enumerate(instances):
                newOptimum[newInstances[i].replace('"', '')] = optimum[ins]
            with open('/home/liuwei/GAST/ACPP/AC_output/GAST/it1/'
                      'TSP_optimum.json', 'w+') as f:
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
            featureFile = '/home/liuwei/GAST/ACPP/AC_output/GAST/it1/whole_instance_feature'
            with open(featureFile, 'w+') as f:
                f.write(featureLineDict['firstLine'] + '\n')
                for i, ins in enumerate(instances):
                    # tempSign = os.path.basename(ins)        ###
                    # insName = ('\"/home/liuwei/GAST/ACPP/AC_output/'
                    #            'GAST/it1/%s\"') % (tempSign)
                    # f.write(insName + ',' + featureLineDict[ins] + '\n')
                    insName = ('\"/home/liuwei/GAST/ACPP/AC_output/'
                               'GAST/it1/%d\"') % (i+1)
                    f.write(insName + ',' + featureLineDict[ins] + '\n')
    currentP, initialInc = comSearch(currentP, initialInc, configurationTime,
                                     validationTime,
                                     cutoffTime, acRuns, instanceIndexFile,
                                     paramFile, featureFile, logFile,
                                     seedHelper)

    logFile.write('End iteration Current solver is \n%s\n' % str(currentP))
    logFile.flush()

    # insGen
    if runNum == algNum:
        break
    instanceIndexFile, featureFile = insGen(currentP, instanceIndexFile, generationTime,
                                            logFile, minTestTimes, maxIt, acRuns,
                                            algNum, paramFile, featureFile, mu, cityUpperB,
                                            cityLowerB, changeP, cutoffTime, retein, newSize,
                                            pSelection, surSelection)
    # print 10086
    # exit(0)
fullSolver = []
for runNum in range(1, algNum + 1):
    fullSolver.append(currentP[runNum].replace('-@1', '-@%d' % (runNum)))
fullSolver = ' '.join(fullSolver)
logFile.write('Final solver:\n%s' % fullSolver)
logFile.close()

