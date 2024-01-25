'''
Instance generation for TSP/SAT
generate new instances based on current portoflio
'''
import time
import sys
import subprocess
import json
import random
import os
import glob
import numpy as np
import psutil

# for feature computing 108 fature in total (including name)
# -instance name 1
# -base 54
# -sp 19
# -dia 6
# -cl 19
# -unit 6
# -lobjois 3
headers = 'instance,nvarsOrig,nclausesOrig,nvars,nclauses,reducedVars,reducedClauses,Pre-featuretime,vars-clauses-ratio,POSNEG-RATIO-CLAUSE-mean,POSNEG-RATIO-CLAUSE-coeff-variation,POSNEG-RATIO-CLAUSE-min,POSNEG-RATIO-CLAUSE-max,POSNEG-RATIO-CLAUSE-entropy,VCG-CLAUSE-mean,VCG-CLAUSE-coeff-variation,VCG-CLAUSE-min,VCG-CLAUSE-max,VCG-CLAUSE-entropy,UNARY,BINARY+,TRINARY+,Basic-featuretime,VCG-VAR-mean,VCG-VAR-coeff-variation,VCG-VAR-min,VCG-VAR-max,VCG-VAR-entropy,POSNEG-RATIO-VAR-mean,POSNEG-RATIO-VAR-stdev,POSNEG-RATIO-VAR-min,POSNEG-RATIO-VAR-max,POSNEG-RATIO-VAR-entropy,HORNY-VAR-mean,HORNY-VAR-coeff-variation,HORNY-VAR-min,HORNY-VAR-max,HORNY-VAR-entropy,horn-clauses-fraction,VG-mean,VG-coeff-variation,VG-min,VG-max,KLB-featuretime,CG-mean,CG-coeff-variation,CG-min,CG-max,CG-entropy,cluster-coeff-mean,cluster-coeff-coeff-variation,cluster-coeff-min,cluster-coeff-max,cluster-coeff-entropy,CG-featuretime,SP-bias-mean,SP-bias-coeff-variation,SP-bias-min,SP-bias-max,SP-bias-q90,SP-bias-q10,SP-bias-q75,SP-bias-q25,SP-bias-q50,SP-unconstraint-mean,SP-unconstraint-coeff-variation,SP-unconstraint-min,SP-unconstraint-max,SP-unconstraint-q90,SP-unconstraint-q10,SP-unconstraint-q75,SP-unconstraint-q25,SP-unconstraint-q50,sp-featuretime,DIAMETER-mean,DIAMETER-coeff-variation,DIAMETER-min,DIAMETER-max,DIAMETER-entropy,DIAMETER-featuretime,cl-num-mean,cl-num-coeff-variation,cl-num-min,cl-num-max,cl-num-q90,cl-num-q10,cl-num-q75,cl-num-q25,cl-num-q50,cl-size-mean,cl-size-coeff-variation,cl-size-min,cl-size-max,cl-size-q90,cl-size-q10,cl-size-q75,cl-size-q25,cl-size-q50,cl-featuretime,vars-reduced-depth-1,vars-reduced-depth-4,vars-reduced-depth-16,vars-reduced-depth-64,vars-reduced-depth-256,unit-featuretime,lobjois-mean-depth-over-vars,lobjois-log-num-nodes-over-vars,lobjois-featuretime'


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
        parentIndex = 0  # only use when pSelection is 'uniform'
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
                parent2 = random.sample(range(0, parent1) + range(parent1 + 1, insLen), 1)[0]
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
            splitPoints = sorted(random.sample(range(1, len(corParent2)), len(corParent1) - 1))
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
                if p <= mu * changeP[0] and currentLen1 > cityLowerB:  # delete this city
                    currentLen1 -= 1
                elif p <= mu * changeP[1] and p > mu * changeP[0] and \
                        currentLen1 < cityUpperB:  # add a city
                    corChild1.append((random.randint(0, 999999), random.randint(0, 999999)))
                    corChild1.append((random.randint(0, 999999), random.randint(0, 999999)))
                    currentLen1 += 1
                else:  # whenever deletion and add both fail, execute pure mutate
                    corChild1.append((random.randint(0, 999999), random.randint(0, 999999)))

            for city in oldCorChild2:
                p = random.random()
                if p > mu:
                    corChild2.append(city)
                    continue
                if p <= mu * changeP[0] and currentLen2 > cityLowerB:  # delete this city
                    currentLen2 -= 1
                elif p <= mu * changeP[1] and p > mu * changeP[0] and \
                        currentLen2 < cityUpperB:  # add a city
                    corChild2.append((random.randint(0, 999999), random.randint(0, 999999)))
                    corChild2.append((random.randint(0, 999999), random.randint(0, 999999)))
                    currentLen2 += 1
                else:  # whenever deletion and add both fail, execute pure mutate
                    corChild2.append((random.randint(0, 999999), random.randint(0, 999999)))
            # if (len(newInstances) + 1) == 6:
            #     print 10086
            #     exit(0)
            # save children as instances, add to newInstances
            insName = ('/home/liuwei/GAST/ACPP/AC_output/'
                       'GAST/it%d/n%d' % (algNum + 1, len(newInstances) + 1))

            f1 = open(insName, 'w+')
            f1.write('NAME : newins-%d\n' % (len(newInstances) + 1))
            f1.write('COMMENT : NONE\n')
            f1.write('TYPE : TSP\n')
            f1.write('DIMENSION : %d\n' % len(corChild1))
            f1.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
            f1.write('NODE_COORD_SECTION\n')
            for i, city in enumerate(corChild1):
                f1.write('%d %d %d\n' % (i + 1, city[0], city[1]))
            f1.close()
            newInstances.append('\"%s\"' % insName)

            if pSelection == 'uniform':
                # in this case only save one child
                continue

            insName = ('/home/liuwei/GAST/ACPP/AC_output/'
                       'GAST/it%d/n%d' % (algNum + 1, len(newInstances) + 1))
            f2 = open(insName, 'w+')
            f2.write('NAME : newins-%d\n' % (len(newInstances) + 1))
            f2.write('COMMENT : NONE\n')
            f2.write('TYPE : TSP\n')
            f2.write('DIMENSION : %d\n' % len(corChild2))
            f2.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
            f2.write('NODE_COORD_SECTION\n')
            for i, city in enumerate(corChild2):
                f2.write('%d %d %d\n' % (i + 1, city[0], city[1]))
            f2.close()
            newInstances.append('\"%s\"' % insName)

        # solve all new instances optimally
        # change dir to confilogs
        os.chdir('/home/liuwei/GAST/ACPP/src/util/configlogs')
        concorde = '/home/liuwei/GAST/ACPP/Solver/Concorde/concorde'
        # run acRuns * k exact solvers in parallel
        subProcess = set()
        for i, ins in enumerate(newInstances):
            cmd = '%s %s > ./qua_n%d' % (concorde, ins, i + 1)
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
            with open('./qua_n%d' % (i + 1), 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'Optimal Solution' in line:
                    solution = line[line.find(':') + 1:].strip()
                    newOptimum[ins.replace('"', '')] = solution
                    break
        cmd = 'rm qua_n*'
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
        # save in newOptimum and TSP_new_optimum.json
        with open(('/home/liuwei/GAST/ACPP/AC_output/GAST/it%d/'
                   'TSP_new_optimum.json' % (algNum + 1)), 'w+') as f:
            json.dump(newOptimum, f)

        os.chdir('/home/liuwei/GAST/ACPP/src/GAST')
        return newInstances, newOptimum
    elif 'lingeling' in paramFile:
        # in SAT mode, call spig.py, always use uniform selection
        # for each ins, randomly select 5 instances as insts

        # First we exclude large instances
        pool = []
        for ins in instances:
            ins = ins.replace('"', "")
            statinfo = os.stat(ins)
            if statinfo.st_size / (1024.0 * 1024.0) > 100:  # extract > 100M instances
                continue
            pool.append(ins)


        spig = 'python /home/liuwei/GAST/ACPP/instance_set/spig/spig.py'
        outputDir = '/home/liuwei/GAST/ACPP/AC_output/GAST/it%d/' % (algNum + 1)
        lingelingCmd = '/home/liuwei/GAST/ACPP/Solver/lingeling-ala-b02aa1a-121013/lingeling'
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
                    prefix = 'base_%d_' % (i + 1)
                    seed = random.randint(1, 1000000)
                    # randomly select poolSize instances as insts
                    insts = random.sample(pool, poolSie)
                    while ins in insts:
                        insts = random.sample(pool, poolSie)
                    cmd = ('%s --seed %d --outputDir %s --outputPrefix %s '
                           '--lingelingCmd %s  --iterations %d %s') % \
                          (spig, seed, outputDir, prefix, lingelingCmd, \
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
            newinsts = glob.glob(outputDir + 'base_%d_*' % (i + 1))
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
    fullSolver = list()
    for i in range(1, algNum + 1):
        fullSolver.append(currentP[i].replace('-@1', '-@%d' % (i)))
    fullSolver = ' '.join(fullSolver)
    running_tasks = 0
    sub_process = set()
    outDir = '/home/liuwei/GAST/ACPP/AC_output/GAST/it%d/' % (algNum + 1)
    for i, ins in enumerate(newInstances):
        for j in range(minTestTimes):
            while True:
                if running_tasks * algNum >= k * acRuns:
                    time.sleep(0.1)
                    finished = [
                        pid for pid in sub_process if pid.poll() is not None]
                    sub_process -= set(finished)
                    running_tasks = len(sub_process)
                    continue
                else:
                    seed = random.randint(0, 1000000)
                    output_file = '%sIns%d_Seed%d' % (outDir, i, j)
                    cmd = ('python /home/liuwei/GAST/ACPP/'
                           'src/util/testing_wrapper.py %s %s %d %d %d %s') % \
                          (ins, output_file, cutoffTime, \
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

            # cmd = 'touch %sIns%d_Seed%d' % (outDir, i, j)
            # process = subprocess.Popen(cmd, shell=True)
            # process.communicate()  ###

            output_file = '%sIns%d_Seed%d' % (outDir, i, j)
            with open(output_file, 'r') as f:
                outPut = f.read().strip()
                values = outPut[outPut.find(':') + 1:].strip().replace(' ', '').split(',')
            # if len(values) == 1:
            #     values.append(0)  ###

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


def insGen(currentP, instanceIndexFile, generationTime,
           logFile, minTestTimes, maxIt, acRuns,
           k, paramFile, featureFile, mu, cityUpperB,
           cityLowerB, changeP, cutoffTime, retein, newSize,
           pSelection, surSelection):
    algNum = len(currentP)
    with open(instanceIndexFile, 'r') as f:
        instances = f.read().strip().split('\n')
    oldInstances = instances
    newInstances = []
    for i, ins in enumerate(instances):
        ins = ins.replace('"', '')
        cmd = ('cp %s /home/liuwei/GAST/ACPP/AC_output/'
               'GAST/it%d/%d' % (ins, algNum + 1, i + 1))
        subprocess.check_output(cmd, shell=True)
        newInstances.append('\"/home/liuwei/GAST/ACPP/AC_output/'
                            'GAST/it%d/%d\"' % (algNum + 1, i + 1))
        # tempSign = os.path.basename(ins)
        # print tempSign
        # cmd = ('cp %s /home/liuwei/GAST/ACPP/AC_output/'
        #        'GAST/it%d/%s' % (ins, algNum + 1, tempSign))
        # subprocess.check_output(cmd, shell=True)
        # newInstances.append('\"/home/liuwei/GAST/ACPP/AC_output/'
        #                     'GAST/it%d/%s\"' % (algNum + 1, tempSign))       ###
    newOptimum = None
    if 'lkh' in paramFile:
        with open(('/home/liuwei/GAST/ACPP/AC_output/GAST/it%d/'
                   'TSP_optimum.json' % algNum), 'r') as f:
            oldOptimum = json.load(f)
        newOptimum = dict()
        for i, ins in enumerate(instances):
            newOptimum[newInstances[i].replace('"', '')] = oldOptimum[ins.replace('"', '')]
        with open(('/home/liuwei/GAST/ACPP/AC_output/GAST/it%d/'
                   'TSP_optimum.json' % (algNum + 1)), 'w+') as f:
            json.dump(newOptimum, f)

    instances = newInstances
    optimum = newOptimum

    logFile.write('-----------------Instance Generation-------------\n')
    logFile.write('Generate new instances in iteration %d\n' % algNum)

    # Get initial fitness, save in performance matrix
    targetF = '/home/liuwei/GAST/ACPP/validation_output/GAST/it%s' % (algNum)
    perM = np.load('%s/performance_matrix.npy' % targetF)
    # print perM
    # exit(0)
    if np.sum(np.isnan(perM)) > 0:
        print 'NAN in perM, error!\n'
        # sys.exit(1)   ###

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
                                               paramFile, algNum, mu,
                                               cityUpperB, cityLowerB, changeP,
                                               pSelection, logFile)
        newInsCount = len(newInstances)
        if newInsCount == 0:
            logFile.write('Failing to generate any new instances, gt next iteration\n\n')
            ite += 1
            continue
        logFile.write('Generated %d new instances, using %s seconds, '
                      'testing..\n\n' % (len(newInstances), str(time.time() - stime)))

        stime = time.time()
        newFitness = testNew(currentP, algNum, newInstances,
                             minTestTimes, paramFile, k, acRuns, cutoffTime)
        logFile.write('Testing done, using %s seconds\n\n' % str(time.time() - stime))

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
                logFile.write('Tournament: %s, %f %f\n' % \
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
            targetFile = ('\"/home/liuwei/GAST/ACPP/AC_output/'
                          'GAST/it%d/%d\"') % (algNum + 1, i + 1)
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
            with open(('/home/liuwei/GAST/ACPP/AC_output/GAST/it%d/'
                       'TSP_optimum.json' % (algNum + 1)), 'w+') as f:
                json.dump(optimum, f)
            cmd = ('rm /home/liuwei/GAST/ACPP/AC_output/GAST/it%d/'
                   'TSP_new_optimum.json' % (algNum + 1))
            subprocess.check_output(cmd, shell=True)

        logFile.write('--------Iteration %d Done---------, Using time %s\n\n' % \
                      (ite, str(time.time() - startingTime)))
        ite += 1

    # build instance_index_file: training_instances
    l = len(instances)
    if retein:
        # first we only randomly select newSize instances to stay
        # other instances will be deleted
        deleteIndex = sorted(random.sample(range(0, l), l - newSize), reverse=True)
        for index in deleteIndex:
            cmd = 'rm %s' % instances[index]
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
            del instances[index]
        tmpOptimum = dict()
        for i, ins in enumerate(instances):
            targetFile = ('\"/home/liuwei/GAST/ACPP/AC_output/'
                          'GAST/it%d/%d\"') % (algNum + 1, i + 1)
            if ins != targetFile:
                cmd = 'mv %s %s' % (ins, targetFile)
                subprocess.check_output(cmd, shell=True)
            instances[i] = targetFile
            if optimum is not None:
                ins = ins.replace('"', '')
                tmpOptimum[instances[i].replace('"', '')] = optimum[ins]
        if optimum is not None:
            optimum = tmpOptimum
            with open(('/home/liuwei/GAST/ACPP/AC_output/GAST/it%d/'
                       'TSP_optimum.json' % (algNum + 1)), 'w+') as f:
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
            cmd = ('cp %s /home/liuwei/GAST/ACPP/AC_output/'
                   'GAST/it%d/%d' % (ins, algNum + 1, l + 1))
            subprocess.check_output(cmd, shell=True)
            instances.append('\"/home/liuwei/GAST/ACPP/AC_output/'
                             'GAST/it%d/%d\"' % (algNum + 1, l + 1))

            if optimum is not None:
                optimum[instances[-1].replace('"', '')] = oldOptimum[ins.replace('"', '')]
            l = l + 1

        if optimum is not None:
            with open(('/home/liuwei/GAST/ACPP/AC_output/GAST/it%d/'
                       'TSP_optimum.json' % (algNum + 1)), 'w+') as f:
                json.dump(optimum, f)

    instanceIndexFile = ('/home/liuwei/GAST/ACPP/AC_output/'
                         'GAST/it%d/training_instances' % (algNum + 1))
    with open(instanceIndexFile, 'w+') as f:
        for ins in instances:
            f.write('%s\n' % ins)

    # For SAT, we need to compute features
    # -base -sp -cl -ls -lobjois
    if 'lingeling' in paramFile:
        # featurePath = ('/home/liuwei/GAST/ACPP/instance_set/SAT/indices/whole_instance_feature')  ###
        featurePath = featureFile  ###
        running_tasks = 0
        sub_process = set()
        options = ['-base', '-sp', '-dia', '-cl', '-unit', '-lobjois']
        featureLenDict = {'-base': 54, '-sp': 19, '-dia': 6, '-cl': 19, '-unit': 6, '-lobjois': 3}
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
                        outFile = ('/home/liuwei/GAST/ACPP/AC_output/'
                                   'GAST/it%d/feature_%d_%s' % \
                                   (algNum + 1, i + 1, option))
                        cmd = '%s %s %s %s' % \
                              (featurePath, option, ins, outFile)
                        chmod = 'chmod 777 %s' % featurePath
                        psutil.Popen(chmod, shell=True)
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
        featureFile = ('/home/liuwei/GAST/ACPP/AC_output/'
                       'GAST/it%d/instance_feature' % (algNum + 1))
        with open(featureFile, 'w+') as f:
            f.write(headers + '\n')
            for i, ins in enumerate(instances):
                outLine = '%s,' % ins
                for option in options:
                    flag = True
                    outFile = ('/home/liuwei/GAST/ACPP/AC_output/'
                               'GAST/it%d/feature'
                               '_%d_%s' % (algNum + 1, i + 1, option))
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
                        outLine += (','.join(['-512.000000000'] * featureLenDict[option]) + ',')
                outLine = outLine.strip(',') + '\n'
                f.write(outLine)

        cmd = 'rm /home/liuwei/GAST/ACPP/AC_output/GAST/it%d/feature_*' % (algNum + 1)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()  ###
    cmd = 'rm /tmp/output*'
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()

    logFile.write('Instance Generation Done at it %d, time %f\n' % \
                  (ite, time.time() - startingTime))
    return instanceIndexFile, featureFile
