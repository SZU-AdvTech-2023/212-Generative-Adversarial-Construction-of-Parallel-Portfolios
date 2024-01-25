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

def generateNew(instances, perM, acRuns, k,
                paramFile, algNum, mu, cityUpperB,
                cityLowerB, changeP, pSelection, logFile):
    # generate newinstances based on current instances
    newInstances = list()
    insLen = len(instances)
    if 'lkh' in paramFile:
        # in TSP mode, generate |insLen| new instances
        # read instances as coordinate forms, each city is
        # [0, 1000000) [0, 1000000)
        insCorList = list()
        for ins in instances:
            insCor = list()
            with open(ins.replace('"', ''), 'r') as f:
                lines = f.read().strip().split('\n')
                lines = lines[6:]
                for line in lines:
                    _, x, y = line.strip().split()
                    insCor.append((int(x), int(y)))
            insCorList.append(insCor)
        parentIndex = 0 #only use when pSelection is 'uniform'
        while len(newInstances) < insLen:
            # 2-tournament selection, select 2 parents
            if pSelection == 'tournament':
                parents = random.sample(range(0, insLen), 4)
                if perM[0, parents[0]] >= perM[0, parents[1]]:
                    parent1 = parents[0]
                else:
                    parent1 = parents[1]
                if perM[0, parents[2]] >= perM[0, parents[3]]:
                    parent2 = parents[2]
                else:
                    parent2 = parents[3]
            elif pSelection == 'uniform':
                parent1 = parentIndex
                parent2 = random.sample(range(0, parent1) + range(parent1+1, insLen), 1)[0]
                parentIndex += 1
                if parentIndex >= insLen:
                    parentIndex = 0
            # crossover
            if len(insCorList[parent1]) <= len(insCorList[parent2]):
                corParent1 = insCorList[parent1]
                corParent2 = insCorList[parent2]
            else:
                corParent1 = insCorList[parent2]
                corParent2 = insCorList[parent1]
            splitPoints = sorted(random.sample(range(1, len(corParent2)), len(corParent1)-1))
            splitPoints.append(len(corParent2))
            corChild1 = list()
            corChild2 = list()
            startingPoint1 = 0
            startingPoint2 = 0
            for point1, point2 in enumerate(splitPoints):
                point1 += 1
                component1 = corParent1[startingPoint1:point1]
                component2 = corParent2[startingPoint2:point2]
                if random.random() <= 0.5:
                    corChild1.extend(component1)
                    corChild2.extend(component2)
                else:
                    corChild1.extend(component2)
                    corChild2.extend(component1)
                startingPoint1 = point1
                startingPoint2 = point2
            # mutation
            oldCorChild1 = corChild1
            oldCorChild2 = corChild2
            currentLen1 = len(oldCorChild1)
            currentLen2 = len(oldCorChild2)
            corChild1 = list()
            corChild2 = list()
            for city in oldCorChild1:
                p = random.random()
                if p > mu:
                    corChild1.append(city)
                    continue
                if p <= mu * changeP[0] and currentLen1 > cityLowerB: # delete this city
                    currentLen1 -= 1
                elif p <= mu * changeP[1] and p > mu * changeP[0] and\
                     currentLen1 < cityUpperB: # add a city
                    corChild1.append((random.randint(0, 999999), random.randint(0, 999999)))
                    corChild1.append((random.randint(0, 999999), random.randint(0, 999999)))
                    currentLen1 += 1
                else: # whenever deletion and add both fail, execute pure mutate
                    corChild1.append((random.randint(0, 999999), random.randint(0, 999999)))

            for city in oldCorChild2:
                p = random.random()
                if p > mu:
                    corChild2.append(city)
                    continue
                if p <= mu * changeP[0] and currentLen2 > cityLowerB: # delete this city
                    currentLen2 -= 1
                elif p <= mu * changeP[1] and p > mu * changeP[0] and\
                     currentLen2 < cityUpperB: # add a city
                    corChild2.append((random.randint(0, 999999), random.randint(0, 999999)))
                    corChild2.append((random.randint(0, 999999), random.randint(0, 999999)))
                    currentLen2 += 1
                else: # whenever deletion and add both fail, execute pure mutate
                    corChild2.append((random.randint(0, 999999), random.randint(0, 999999)))

            # save children as instances, add to newInstances
            insName = (os.path.abspath(os.path.realpath(__file__)+"/../../..")+
                       '/AC_output/GAST/it%d/n%d' % (algNum+1, len(newInstances)+1))
            f1 = open(insName, 'w+')
            f1.write('NAME : newins-%d\n' % (len(newInstances)+1))
            f1.write('COMMENT : NONE\n')
            f1.write('TYPE : TSP\n')
            f1.write('DIMENSION : %d\n' % len(corChild1))
            f1.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
            f1.write('NODE_COORD_SECTION\n')
            for i, city in enumerate(corChild1):
                f1.write('%d %d %d\n' % (i+1, city[0], city[1]))
            f1.close()
            newInstances.append('\"%s\"' % insName)

            if pSelection == 'uniform':
                # in this case only save one child
                continue

            insName = (os.path.abspath(os.path.realpath(__file__)+"/../../..")+
                       '/AC_output/GAST/it%d/n%d' % (algNum+1, len(newInstances)+1))
            f2 = open(insName, 'w+')
            f2.write('NAME : newins-%d\n' % (len(newInstances)+1))
            f2.write('COMMENT : NONE\n')
            f2.write('TYPE : TSP\n')
            f2.write('DIMENSION : %d\n' % len(corChild2))
            f2.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
            f2.write('NODE_COORD_SECTION\n')
            for i, city in enumerate(corChild2):
                f2.write('%d %d %d\n' % (i+1, city[0], city[1]))
            f2.close()
            newInstances.append('\"%s\"' % insName)

        # solve all new instances optimally
        # change dir to confilogs
        os.chdir(os.path.abspath(os.path.realpath(__file__)+"/../../../")+'/src/util/configlogs')
        concorde = os.path.abspath(os.path.realpath(__file__)+"/../../../../")+'/Solver/Concorde/concorde'
        # run acRuns * k exact solvers in parallel
        subProcess = set()
        for i, ins in enumerate(newInstances):
            cmd = '%s %s > ./qua_n%d' % (concorde, ins, i+1)
            subProcess.add(subprocess.Popen(cmd, shell=True))
        while subProcess:
            time.sleep(5)
            print 'Still %d solving process not exits' % len(subProcess)
            finished = [pid for pid in subProcess if pid.poll() is not None]
            subProcess -= set(finished)
        # clear configlogs
        cmd = 'rm *.mas *.pul *.sav *.sol *.res'
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
        # read quality file
        newOptimum = dict()
        for i, ins in enumerate(newInstances):
            with open('./qua_n%d' % (i+1), 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'Optimal Solution' in line:
                    solution = line[line.find(':')+1:].strip()
                    newOptimum[ins.replace('"', '')] = solution
                    break
        cmd = 'rm qua_n*'
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
        # save in newOptimum and TSP_new_optimum.json
        with open((os.path.abspath(os.path.realpath(__file__)+"/../../../../")+'/AC_output/GAST/it%d/'
                   'TSP_new_optimum.json' % (algNum+1)), 'w+') as f:
            json.dump(newOptimum, f)

        os.chdir(os.path.abspath(os.path.realpath(__file__)+"/../../../../")+'/src/GAST')
        return newInstances, newOptimum
    elif 'lingeling' in paramFile:
        # in SAT mode, call spig.py, always use uniform selection
        # for each ins, randpmly select 5 instances as insts

        # First we exclude large instances
        pool = []
        for ins in instances:
            ins = ins.replace('"', "")
            statinfo = os.stat(ins)
            if statinfo.st_size / (1024.0 * 1024.0) > 100:  # extract > 100M instances
                continue
            pool.append(ins)

        spig = 'python /home/liusc/projects/ACPP/instance_set/spig/spig.py'
        outputDir = '/home/liusc/projects/ACPP/AC_output/GAST/it%d/' % (algNum+1)
        lingelingCmd = '/home/liusc/projects/ACPP/Solver/lingeling-ars/lingeling'
        iteNum = 5
        poolSie = 7

        sub_process = set()
        running_tasks = 0
        maxPar = 49
        for i, ins in enumerate(instances):
            ins = ins.replace('"', '')
            if ins not in pool:
                continue
            while True:
                if running_tasks >= maxPar:
                    time.sleep(5)
                    finished = [pid for pid in sub_process if pid.poll() is not None]
                    sub_process -= set(finished)
                    running_tasks = len(sub_process)
                    continue
                else:
                    prefix = 'base_%d_' % (i+1)
                    seed = random.randint(1, 1000000)
                    # randomly select poolSize instances as insts
                    insts = random.sample(pool, poolSie)
                    while ins in insts:
                        insts = random.sample(pool, poolSie)
                    cmd = ('%s --seed %d --outputDir %s --outputPrefix %s '
                           '--lingelingCmd %s  --iterations %d %s') %\
                          (spig, seed, outputDir, prefix, lingelingCmd,\
                           iteNum, ins)
                    for ref_ins in insts:
                        cmd = cmd + ' %s' % ref_ins
                    pid = psutil.Popen(cmd, shell=True)
                    # if idleCpu is None:
                    #     idleCpu = pid.cpu_affinity()
                    # pid.cpu_affinity([idleCpu[0]])
                    # idleCpu.pop(0)
                    # if not idleCpu:
                    #     idleCpu = None
                    # logFile.write('ins: %d, assigned cpu: %s\n' % (i+1, str(pid.cpu_affinity())))
                    sub_process.add(pid)
                    running_tasks = len(sub_process)
                    break

        # check if subprocess all exits
        while sub_process:
            time.sleep(5)
            print 'Still %d spig process not exits' % len(sub_process)
            finished = [pid for pid in sub_process if pid.poll() is not None]
            sub_process -= set(finished)

        newInstances = []
        # only save one ins generated based on each seed ins
        for i, _ in enumerate(instances):
            newinsts = glob.glob(outputDir+'base_%d_*' % (i+1))
            if newinsts:
                newInstances.append(random.choice(newinsts))
                for ins in newinsts:
                    if ins != newInstances[-1]:
                        cmd = 'rm %s' % ins
                        p = subprocess.Popen(cmd, shell=True)
                        p.communicate()

        for i, ins in enumerate(newInstances):
            newInstances[i] = '\"%s\"' % ins
        return newInstances, None

def testNew(currentP, algNum, newInstances,
            minTestTimes, paramFile, k, acRuns, cutoffTime):
    #print("begin testNew")
    fullSolver = list()
    for i in range(1, algNum+1):
        fullSolver.append(currentP[i].replace('-@1', '-@%d' % (i)))
    fullSolver = ' '.join(fullSolver)
    running_tasks = 0
    sub_process = set()
    outDir = os.path.abspath(os.path.realpath(__file__)+"/../../..")+'/AC_output/GAST/it%d/' % (algNum+1)
    #print(enumerate(newInstances))
    for i, ins in enumerate(newInstances):
        for j in range(minTestTimes):
            while True:
                #print(("%d,%d,%s,%s")%(i,j,str(running_tasks*algNum),str(k*acRuns))) 
                if running_tasks * algNum >= 4 * acRuns:
                    time.sleep(0.1)
                    finished = [
                        pid for pid in sub_process if pid.poll() is not None]
                    sub_process -= set(finished)
                    running_tasks = len(sub_process)
                    continue
                else:
                    seed = random.randint(0, 1000000)
                    output_file = '%sIns%d_Seed%d' % (outDir, i, j)
                    cmd = ('python '+os.path.abspath(os.path.realpath(__file__)+"/../../..")+
                           '/src/util/testing_wrapper.py %s %s %d %d %d %s') %\
                          (ins, output_file, cutoffTime,\
                           0, seed, fullSolver)
                    sub_process.add(psutil.Popen(cmd, shell=True))
                    running_tasks = len(sub_process)
                    break

    # check if subprocess all exits
    while sub_process:
        time.sleep(5)
        print 'Still %d testing-instance process not exits' % len(sub_process)
        finished = [pid for pid in sub_process if pid.poll() is not None]
        sub_process -= set(finished)

    # extract testing results
    punish = 10
    # performance matrix, [i,j] i+1 run j ins
    newFitness = np.zeros((1, len(newInstances))) * np.nan
    runCount = np.zeros(newFitness.shape) * np.nan
    for i, _ in enumerate(newInstances):
        for j in range(minTestTimes):
            output_file = '%sIns%d_Seed%d' % (outDir, i, j)
            if os.path.isfile(output_file):
                with open(output_file, 'r') as f:
                    outPut = f.read().strip()
                    values = outPut[outPut.find(':') + 1:].strip().replace(' ', '').split(',')
                (result, runtime) = (values[0], float(values[1]))
                if 'TIMEOUT' in result:
                    runtime = runtime * punish
                if np.isnan(newFitness[0, i]):
                    newFitness[0, i] = runtime
                    runCount[0, i] = 1
                else:
                    newFitness[0, i] += runtime
                    runCount[0, i] += 1
    newFitness = np.true_divide(newFitness, runCount)
    # clear dir
    cmd = 'rm %sIns*' % outDir
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    return newFitness



#realpath,dirname,
file_path=os.path.realpath(__file__)
file_path=os.path.dirname(file_path)
file_path=os.path.abspath(file_path+"/../..")
#os.path.abspath(os.path.realpath(__file__)+"/../../..")#ACPP path



# Set parameter file and algorithm number
#paramFile = '/home/liusc/projects/ACPP/Solver/paramfile/Single_lingeling_ala_pcs.txt'
paramFile = os.path.abspath(file_path+'/Solver/paramfile/Single_lkh_pcs.txt')
algNum = 8 
# Set initial training instance index file
domain = 'TSP'
mode = "small"
expNum = 1


instanceIndexFile = os.path.abspath(file_path+'/instance_set/%s/indices/training_index_%s_%d' %\
                    (domain, mode, expNum))
featureFile = os.path.abspath(file_path+'/instance_set/SAT/indices/whole_instance_feature')
if domain == 'TSP':
    featureFile = None
    configurationTime = 5400#3600*1.5
    validationTime = 1800#3600*0.5
    generationTime = 18000#3600*5
    cutoffTime=1 #TSP

if domain == 'SAT':
# Set time options for SAT
    configurationTime = 3600*8
    validationTime = 3600*4
    generationTime = 3600*4
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
surSelection = 'tournament'
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
logFile = open("GAST_log.txt", "w+")
initialInc = []

cmd = 'rm -r '+os.path.abspath(file_path+'/validation_output/GAST/it*')
p = subprocess.Popen(cmd, shell=True)
p.communicate()
cmd = 'rm -r '+os.path.abspath(file_path+'/AC_output/GAST/it*')
p = subprocess.Popen(cmd, shell=True)
p.communicate()

for runNum in range(1, algNum+1):
    cmd = 'mkdir '+os.path.abspath(file_path+'/AC_output/GAST/it%d' % runNum)
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

for runNum_iter in range(1, algNum+1):
    logFile.write('-------------------------------------------\n')
    logFile.write('---------------Iteration %d ---------------\n' % runNum_iter)
    if runNum_iter == 1:
        cmd = ('cp %s ')% instanceIndexFile+os.path.abspath(file_path+'/AC_output/GAST/it1/training_instances')
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

        instanceIndexFile = file_path+'/AC_output/GAST/it1/training_instances'
        with open(instanceIndexFile, 'r') as f:
            instances = f.read().strip().replace('"', '').split('\n')
        newInstances = []
        for i, ins in enumerate(instances):
            cmd = ('cp %s '%ins+os.path.abspath(file_path+'/AC_output/GAST/it1/%d' % (i+1)))
            subprocess.check_output(cmd, shell=True)
            newInstances.append('\"'+os.path.abspath(file_path+'/AC_output/GAST/it1/%d'% (i+1))+'\"')
        with open(instanceIndexFile, 'w+') as f:
            for ins in newInstances:
                f.write(ins + '\n')

        if 'lkh' in paramFile:
            with open(os.path.abspath(file_path+'/instance_set/TSP/indices/TSP_optimum.json'), 'r') as f:
                optimum = json.load(f)
            newOptimum = dict()
            for i, ins in enumerate(instances):
                newOptimum[newInstances[i].replace('"', '')] = optimum[ins]
            with open(os.path.abspath(file_path+'/AC_output/GAST/it1/TSP_optimum.json'), 'w+') as f:
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
            featureFile = os.path.abspath(file_path+'/AC_output/GAST/it1/instance_feature')
            with open(featureFile, 'w+') as f:
                f.write(featureLineDict['firstLine'] + '\n')
                for i, ins in enumerate(instances):
                    insName = '\"'+os.path.abspath(file_path+'/AC_output/GAST/it1/%d' % (i+1))+'\"'
                    f.write(insName + ',' + featureLineDict[ins] + '\n')
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
        cmd1 = "rm -r "+file_path+"/AC_output/GAST/run" + \
            str(runNum) + "/output"
        cmd2 = "mkdir "+file_path+"/AC_output/GAST/run" + \
            str(runNum) + "/output"
        #cmd3 = "cp "+file_path+'/src/GAST/GAST_solver.py '+file_path+('/AC_output/GAST/run%d'
        #               '/GAST_solver.py') % runNum
        tmp = subprocess.Popen(cmd1, shell=True)
        tmp.communicate()
        tmp = subprocess.Popen(cmd2, shell=True)
        tmp.communicate()
        #tmp = subprocess.Popen(cmd3, shell=True)
        #tmp.communicate()

    ######################construct the scenario file for smac
    #con_scenario_file(runs, cutoffTime, instanceIndexFile, featureFile, paramFile)
    training = instanceIndexFile
    testing = training
    for run_number in runs:
        scenarioFile = file_path+('/AC_output/GAST/run%d'
                        '/scenario.txt') % run_number
        FILE = open(scenarioFile, "w+")
        lines = []
        lines.append(('algo = '+file_path+'/src/GAST/GAST_solver.py'+'\n'))
        #lines.append('algo = '+file_path+('/AC_output/GAST/run%d'
        #               '/GAST_solver.py\n') % run_number)
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
        lines.append('outdir = '+file_path+('/AC_output/GAST/run%d/output') %run_number)

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
    with open('exsiting_solver.txt', 'w+') as f:
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
                  file_path+('/AC_output/GAST/run%d'
                   '/scenario.txt') % run_number +\
                  " --wallclock-limit " + \
                  str(Timeout) + " --seed " + str(seedList[i]) + \
                  " --validation false " + \
                  " --console-log-level OFF" + \
                  " --log-level TRACE" + \
                  " --initial-incumbent " + '"' + initialInc + ' "'
        else:
            cmd = "./smac " + " --scenario-file " +\
                file_path+('/AC_output/GAST/run%d/scenario.txt') % run_number +\
                " --wallclock-limit " + \
                str(Timeout) + " --seed " + str(seedList[i]) + \
                " --validation false " + \
                " --console-log-level OFF" + \
                " --log-level TRACE"
        # print cmd
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
        outputDir = glob.glob(file_path+("/AC_output/GAST/run%d/output/run%d/log-run*.txt") %
                              (run, run))[0]
        with open(outputDir, "r") as FILE:
            lines = FILE.read().strip()
            lines = lines[lines.find('has finished'):]
            lines = lines[lines.find('-@1'):]
            configs[run] = lines.split('\n')[0]

        outputDir = glob.glob(file_path+("/AC_output/GAST/run%d/output/run%d/detailed-traj-run-*.csv") %
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
        cmd = ('rm -r '+file_path+'/validation_output/GAST/run%d*' % i)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
    processes = set()
    logFile.write('------Current we have %d Algs-------\n' % (algNumc+1))

    runs = range(1, (acRuns + 1))
    logFile.write('Executing %s runs\n' % str(runs))
    logFile.flush()
    for runNum in runs:
        cmd = 'python GAST_validation.py %s %d %d %d %d %s' %\
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
    outputdir = file_path+'/validation_output/GAST/'
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
    np.save(file_path+('/validation_output/GAST/performance_matrix.npy'), incRow)
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
    outputdir = glob.glob(file_path+"/AC_output/GAST/run%d/output/run%d/log-run*.txt" %
                          (incIndex, incIndex))[0]
    with open(outputdir, "r") as FILE:
        lines = FILE.read().strip()
        lines = lines[lines.find('has finished'):]
        lines = lines[lines.find('-@1'):]
        solver = lines.split('\n')[0]

    result_file = file_path+'/validation_output/GAST/validation_results.txt'
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
    targetF = file_path+'/validation_output/GAST/it%s' % (algNumc+1)
    cmd = 'mkdir %s' % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

    cmd = 'mv '+file_path+'/validation_output/GAST/run'+' %s/' % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    cmd = 'mv '+file_path+'/validation_output/GAST/validation_results.txt %s/' % targetF
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    cmd = 'mv '+file_path+'/validation_output/GAST/performance_matrix.npy %s/' % targetF
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
    # insGen
    if runNum_iter == algNum:
        break
    
    
    
    
    





















    ############################################begin generate instances
    #instanceIndexFile, featureFile = insGen(currentP, instanceIndexFile, generationTime,
    #                                        logFile, minTestTimes, maxIt, acRuns,
    #                                        algNum, paramFile, featureFile, mu, cityUpperB,
    #                                        cityLowerB, changeP, cutoffTime, retein, newSize,
    #                                        pSelection, surSelection)
    k=algNum
    # for feature computing 108 fature in total (including name)
    # -instance name 1
    # -base 54
    # -sp 19
    # -dia 6
    # -cl 19
    # -unit 6
    # -lobjois 3
    headers = 'instance,nvarsOrig,nclausesOrig,nvars,nclauses,reducedVars,reducedClauses,Pre-featuretime,vars-clauses-ratio,POSNEG-RATIO-CLAUSE-mean,POSNEG-RATIO-CLAUSE-coeff-variation,POSNEG-RATIO-CLAUSE-min,POSNEG-RATIO-CLAUSE-max,POSNEG-RATIO-CLAUSE-entropy,VCG-CLAUSE-mean,VCG-CLAUSE-coeff-variation,VCG-CLAUSE-min,VCG-CLAUSE-max,VCG-CLAUSE-entropy,UNARY,BINARY+,TRINARY+,Basic-featuretime,VCG-VAR-mean,VCG-VAR-coeff-variation,VCG-VAR-min,VCG-VAR-max,VCG-VAR-entropy,POSNEG-RATIO-VAR-mean,POSNEG-RATIO-VAR-stdev,POSNEG-RATIO-VAR-min,POSNEG-RATIO-VAR-max,POSNEG-RATIO-VAR-entropy,HORNY-VAR-mean,HORNY-VAR-coeff-variation,HORNY-VAR-min,HORNY-VAR-max,HORNY-VAR-entropy,horn-clauses-fraction,VG-mean,VG-coeff-variation,VG-min,VG-max,KLB-featuretime,CG-mean,CG-coeff-variation,CG-min,CG-max,CG-entropy,cluster-coeff-mean,cluster-coeff-coeff-variation,cluster-coeff-min,cluster-coeff-max,cluster-coeff-entropy,CG-featuretime,SP-bias-mean,SP-bias-coeff-variation,SP-bias-min,SP-bias-max,SP-bias-q90,SP-bias-q10,SP-bias-q75,SP-bias-q25,SP-bias-q50,SP-unconstraint-mean,SP-unconstraint-coeff-variation,SP-unconstraint-min,SP-unconstraint-max,SP-unconstraint-q90,SP-unconstraint-q10,SP-unconstraint-q75,SP-unconstraint-q25,SP-unconstraint-q50,sp-featuretime,DIAMETER-mean,DIAMETER-coeff-variation,DIAMETER-min,DIAMETER-max,DIAMETER-entropy,DIAMETER-featuretime,cl-num-mean,cl-num-coeff-variation,cl-num-min,cl-num-max,cl-num-q90,cl-num-q10,cl-num-q75,cl-num-q25,cl-num-q50,cl-size-mean,cl-size-coeff-variation,cl-size-min,cl-size-max,cl-size-q90,cl-size-q10,cl-size-q75,cl-size-q25,cl-size-q50,cl-featuretime,vars-reduced-depth-1,vars-reduced-depth-4,vars-reduced-depth-16,vars-reduced-depth-64,vars-reduced-depth-256,unit-featuretime,lobjois-mean-depth-over-vars,lobjois-log-num-nodes-over-vars,lobjois-featuretime'
    algNumc = len(currentP)
    with open(instanceIndexFile, 'r') as f:
        instances = f.read().strip().split('\n')
    oldInstances = instances
    newInstances = []
    for i, ins in enumerate(instances):
        cmd = ('cp %s '%ins + file_path+'/AC_output/GAST/it%d/%d' % (algNumc+1, i+1))
        subprocess.check_output(cmd, shell=True)
        newInstances.append('\"'+file_path+'/AC_output/GAST/it%d/%d\"' % (algNumc+1, i+1))
    newOptimum = None
    if 'lkh' in paramFile:
        with open(file_path+'/AC_output/GAST/it%d/TSP_optimum.json' % algNumc, 'r') as f:
            oldOptimum = json.load(f)
        newOptimum = dict()
        for i, ins in enumerate(instances):
            newOptimum[newInstances[i].replace('"', '')] = oldOptimum[ins.replace('"', '')]
        with open(file_path+'/AC_output/GAST/it%d/TSP_optimum.json' % (algNumc+1), 'w+') as f:
            json.dump(newOptimum, f)

    instances = newInstances
    optimum = newOptimum

    logFile.write('-----------------Instance Generation-------------\n')
    logFile.write('Generate new instances in iteration %d\n' % algNumc)

    # Get initial fitness, save in performance matrix
    targetF = file_path+'/validation_output/GAST/it%s' % (algNumc)
    perM = np.load('%s/performance_matrix.npy' % targetF)
    # if np.sum(np.isnan(perM)) > 0:
    #     print 'NAN in perM, error!\n'
    #     sys.exit(1)

    logFile.write('Initial Fitness:\n')
    perM = perM.reshape((1, perM.shape[0]))
    for i in range(len(instances)):
        logFile.write(str(perM[0, i]) + ' ')
    logFile.write('\n')
    logFile.write('Mean: ' + str(np.mean(perM)) + ' \n')

    # generate instances
    ite = 1
    startingTime = time.time()
    while ite <= maxIt and time.time() - startingTime <= generationTime:
        logFile.write('--------Instance Generation Iteration %d---------\n\n' % ite)
        stime = time.time()
        newInstances, newOptimum = generateNew(instances, perM, acRuns, k,
                                               paramFile, algNumc, mu,
                                               cityUpperB, cityLowerB, changeP,
                                               pSelection, logFile)
        newInsCount = len(newInstances)
        if newInsCount == 0:
            logFile.write('Failing to generate any new instances, gt next iteration\n\n')
            ite += 1
            continue
        logFile.write('Generated %d new instances, using %s seconds, '
                      'testing..\n\n' % (len(newInstances), str(time.time()-stime)))
        logFile.flush()
        stime = time.time()
        newFitness = testNew(currentP, algNumc, newInstances,
                             minTestTimes, paramFile, k, acRuns, cutoffTime)
        logFile.write('Testing done, using %s seconds\n\n' % str(time.time()-stime))
        logFile.flush()
        logFile.write('Fitness of new instances:\n\n')
        for i in range(len(newInstances)):
            logFile.write(str(newFitness[0, i]) + ' ')
        logFile.write('\n')

        instances = instances + newInstances
        perM = np.hstack((perM, newFitness))

        logFile.write('Fitness of all instances:\n')
        for i in range(len(instances)):
            logFile.write(str(perM[0, i]) + ' ')
        logFile.write('\n')

        if surSelection == 'truncation':
            # remove the instances with the worst fitness
            # first we sort performance matrix
            logFile.write('Truncation survivior selection...\n')
            sortIndex = np.argsort(perM)
            logFile.write('Sort all instances, results:\n')
            for i in range(len(instances)):
                logFile.write(str(sortIndex[0, i]) + ' ')
            logFile.write('\n')
            # for instances, del index from high to low
            deleteIndex = np.sort(sortIndex[0, 0:newInsCount])
            deleteIndex = deleteIndex[::-1]
        elif surSelection == 'tournament':
            # binary tournament to decide, introducing more diversity
            logFile.write('Tournament survivior selection...\n')
            poolSize = len(instances)
            pool = set(range(0, poolSize))
            selectedIndex = set()
            while len(selectedIndex) < poolSize - newInsCount:
                candidates = random.sample(pool, 2)
                logFile.write('Tournament: %s, %f %f\n' %\
                              (str(candidates), perM[0, candidates[0]], perM[0, candidates[1]]))
                if perM[0, candidates[0]] >= perM[0, candidates[1]]:
                    selectedIndex.add(candidates[0])
                    pool.remove(candidates[0])
                else:
                    selectedIndex.add(candidates[1])
                    pool.remove(candidates[1])
            deleteIndex = np.sort(list(set(range(0, poolSize)) - selectedIndex))
            deleteIndex = deleteIndex[::-1]

        logFile.write('Delete Index:\n')
        for insIndex in deleteIndex:
            logFile.write('%d ' % insIndex)
            cmd = 'rm %s' % instances[insIndex]
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
            del instances[insIndex]
        logFile.write('\n')

        perM = np.delete(perM, deleteIndex, 1)
        logFile.write('Iteration end, perM results: ')
        for i in range(len(instances)):
            logFile.write(str(perM[0, i]) + ' ')
        logFile.write('\n')
        logFile.write('Mean: ' + str(np.mean(perM)) + ' \n')

        # rearrange instances and optimum file (if necessary)
        tmpOptimum = dict()
        for i, ins in enumerate(instances):
            targetFile = '\"'+file_path+'/AC_output/GAST/it%d/%d\"' % (algNumc+1, i+1)
            if ins != targetFile:
                cmd = 'mv %s %s' % (ins, targetFile)
                subprocess.check_output(cmd, shell=True)
            instances[i] = targetFile
            if optimum is not None:
                ins = ins.replace('"', '')
                if ins in optimum:
                    tmpOptimum[instances[i].replace('"', '')] = optimum[ins]
                elif ins in newOptimum:
                    tmpOptimum[instances[i].replace('"', '')] = newOptimum[ins]
                else:
                    print 'error, %s not in any optimum\n' % ins
                    sys.exit(1)
        if optimum is not None:
            optimum = tmpOptimum
            with open(file_path+'/AC_output/GAST/it%d/TSP_optimum.json' % (algNumc+1), 'w+') as f:
                json.dump(optimum, f)
            cmd = ('rm '+file_path+'/AC_output/GAST/it%d/TSP_new_optimum.json' % (algNumc+1))
            subprocess.check_output(cmd, shell=True)

        logFile.write('--------Iteration %d Done---------, Using time %s\n\n' %\
                      (ite, str(time.time()-startingTime)))
        ite += 1

    # build instance_index_file: training_instances
    l = len(instances)
    if retein:
        # first we only randomly select newSize instances to stay
        # other instances will be deleted
        deleteIndex = sorted(random.sample(range(0, l), l-newSize), reverse=True)
        for index in deleteIndex:
            cmd = 'rm %s' % instances[index]
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
            del instances[index]
        tmpOptimum = dict()
        for i, ins in enumerate(instances):
            targetFile = '\"'+file_path+'/AC_output/GAST/it%d/%d\"' % (algNumc+1, i+1)
            if ins != targetFile:
                cmd = 'mv %s %s' % (ins, targetFile)
                subprocess.check_output(cmd, shell=True)
            instances[i] = targetFile
            if optimum is not None:
                ins = ins.replace('"', '')
                tmpOptimum[instances[i].replace('"', '')] = optimum[ins]
        if optimum is not None:
            optimum = tmpOptimum
            with open(file_path+'/AC_output/GAST/it%d/TSP_optimum.json' % (algNumc+1), 'w+') as f:
                json.dump(optimum, f)

        l = newSize
        # use md5 to prevent duplicate
        md5set = set()
        for ins in instances:
            cmd = 'md5sum %s' % ins
            outStr = subprocess.check_output(cmd, shell=True)
            md5set.add(outStr.split()[0])
        # restore old training instances and optimum file
        for ins in oldInstances:
            cmd = 'md5sum %s' % ins
            outStr = subprocess.check_output(cmd, shell=True)
            if outStr.split()[0] in md5set:
                continue
            cmd = ('cp %s '%ins+file_path+'/AC_output/GAST/it%d/%d'%(algNumc+1, l+1))
            subprocess.check_output(cmd, shell=True)
            instances.append('\"'+file_path+'/AC_output/GAST/it%d/%d\"' % (algNumc+1, l+1))

            if optimum is not None:
                optimum[instances[-1].replace('"', '')] = oldOptimum[ins.replace('"', '')]
            l = l + 1

        if optimum is not None:
            with open(file_path+'/AC_output/GAST/it%d/TSP_optimum.json' % (algNumc+1), 'w+') as f:
                json.dump(optimum, f)

    instanceIndexFile = file_path+'/AC_output/GAST/it%d/training_instances' % (algNumc+1)
    with open(instanceIndexFile, 'w+') as f:
        for ins in instances:
            f.write('%s\n' % ins)

    # For SAT, we need to compute features
    # -base -sp -cl -ls -lobjois
    if 'lingeling' in paramFile:
        featurePath = file_path+'/instance_set/feature/SAT-features-competition2012/features'
        running_tasks = 0
        sub_process = set()
        options = ['-base', '-sp', '-dia', '-cl', '-unit', '-lobjois']
        featureLenDict = {'-base':54, '-sp':19, '-dia':6, '-cl':19, '-unit':6, '-lobjois':3}
        for i, ins in enumerate(instances):
            for option in options:
                while True:
                    if running_tasks >= k * acRuns:
                        time.sleep(0.1)
                        finished = [pid for pid in sub_process if pid.poll() is not None]
                        sub_process -= set(finished)
                        running_tasks = len(sub_process)
                        continue
                    else:
                        outFile = file_path+'/AC_output/GAST/it%d/feature_%d_%s' %\
                                  (algNumc+1, i+1, option)
                        cmd = '%s %s %s %s' %\
                            (featurePath, option, ins, outFile)
                        sub_process.add(psutil.Popen(cmd, shell=True))
                        running_tasks = len(sub_process)
                        break

        # check if subprocess all exits
        startFtime = time.time()
        while sub_process:
            time.sleep(1)
            print 'Still %d feature-computing process not exits' % len(sub_process)
            finished = [pid for pid in sub_process if pid.poll() is not None]
            sub_process -= set(finished)
            print 'Has cost %f seconds\n' % (time.time() - startFtime)
            if time.time() - startFtime > 300:
                print 'Kill all feature processes'
                for pid in sub_process:
                    try:
                        pid.kill()
                    except psutil.NoSuchProcess:
                        pass

        # write to instance_feature
        featureFile = file_path+'/AC_output/GAST/it%d/instance_feature' % (algNumc+1)
        with open(featureFile, 'w+') as f:
            f.write(headers + '\n')
            for i, ins in enumerate(instances):
                outLine = '%s,' % ins
                for option in options:
                    flag = True
                    outFile = file_path+'/AC_output/GAST/it%d/feature_%d_%s' % (algNumc+1, i+1, option)
                    if os.path.isfile(outFile):
                        with open(outFile, 'r') as sf:
                            lines = sf.read().strip().split('\n')
                            if len(lines) != 2:
                                flag = False
                            else:
                                outLine += (lines[1] + ',')
                    else:
                        flag = False
                    if not flag:
                        outLine += (','.join(['-512.000000000']*featureLenDict[option]) + ',')
                outLine = outLine.strip(',') + '\n'
                f.write(outLine)

        cmd = 'rm '+file_path+'/AC_output/GAST/it%d/feature_*' % (algNumc+1)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
    cmd = 'rm /tmp/output*'
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

    logFile.write('Instance Generation Done at it %d, time %f\n' %\
                  (ite, time.time()-startingTime))
    #return instanceIndexFile, featureFile
    ############################################end generate instances


fullSolver = []
for runNum in range(1, algNum + 1):
    fullSolver.append(currentP[runNum].replace('-@1', '-@%d' % (runNum)))
fullSolver = ' '.join(fullSolver)
logFile.write('Final solver:\n%s' % fullSolver)
logFile.close()
