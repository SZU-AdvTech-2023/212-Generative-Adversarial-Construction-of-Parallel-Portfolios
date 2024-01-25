import os
import subprocess
import time
import glob
import numpy as np


def validation(insFile, budget, cutoffTime,
               algNum, acRuns, existingSolver,
               logFile):
    for i in range(1, acRuns+1):
        cmd = ('rm -r /home/liuwei/GAST/ACPP/'
               'validation_output/GAST/run%d*' % i)
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
    outputdir = ('/home/liuwei/GAST/ACPP/'
                 'validation_output/GAST/')
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
            if np.isnan(perM[run_number - 1, ins_index]):
                perM[run_number - 1, ins_index] = runtime
                runCount[run_number - 1, ins_index] = 1
            else:
                perM[run_number - 1, ins_index] += runtime
                runCount[run_number - 1, ins_index] += 1

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

    np.save(('/home/liuwei/GAST/ACPP/'
             'validation_output/GAST/performance_matrix.npy'), incRow)
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
    outputdir = glob.glob("/home/liuwei/GAST/ACPP/AC_output/GAST"
                          "/run%d/output/run%d/log-run*.txt" %
                          (incIndex, incIndex))[0]
    with open(outputdir, "r") as FILE:
        lines = FILE.read().strip()
        lines = lines[lines.find('has finished'):]
        lines = lines[lines.find('-@1'):]
        solver = lines.split('\n')[0]

    result_file = ('/home/liuwei/GAST/ACPP/'
                   'validation_output/GAST/validation_results.txt')
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
