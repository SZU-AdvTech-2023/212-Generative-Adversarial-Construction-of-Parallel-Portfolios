import os
import time
import sys
import glob
import psutil


def start_validation(sol, instances,
                     seeds, cutoffTime, budget,
                     outdir, cutoff_length=0):
    beginTime = time.time()
    insIndex = 0

    while True:
        instance = instances[insIndex]
        seed = int(seeds.pop(0))
        output_file = '%s_Ins%d_Seed%d' % (outdir, insIndex, seed)   ###
        cmd = 'python '+os.path.abspath(os.path.realpath(__file__)+"/../../../")+('/src/util/validation_wrapper.py %s %s %d %d %d %s') % (instance, output_file, cutoffTime, cutoff_length, seed, sol)
        PID = psutil.Popen(cmd, shell=True)
        stdOut = PID.communicate()[0]
        #print stdOut
        if (time.time() - beginTime) > budget:
            break
        insIndex += 1
        if insIndex == len(instances):
            insIndex = 0


if __name__ == "__main__":
    file_path=os.path.realpath(__file__)
    file_path=os.path.dirname(file_path)
    file_path=os.path.abspath(file_path+"/../..")
    vInsIndex = sys.argv[1]
    exsitingSolver = sys.argv[6:]
    algNum = int(sys.argv[5])
    with open(vInsIndex, "r") as FILE:
        instance_list = FILE.read().strip().split('\n')
    seed_index_file = (file_path+"/validation_output/GAST/seed_index.txt")
    with open(seed_index_file, "r") as FILE:
        seed_list = FILE.read().strip().split()
    #print(seed_list)
    # set algorithm
    run_number = int(sys.argv[4])
    outputdir = glob.glob((file_path+
                           "/AC_output/GAST/run%d/output/run%d/"
                           "log-run*.txt") %
                          (run_number, run_number))[0]
    with open(outputdir, "r") as FILE:
        lines = FILE.read().strip()
        lines = lines[lines.find('has finished'):]
        lines = lines[lines.find('-@1'):]
        solver = lines.split('\n')[0]
        solver = solver.replace('-@1', '-@%d' % (algNum+1))
    solver = ' '.join(exsitingSolver) + ' ' + solver

    # set other options
    cutoff_time = int(sys.argv[3])
    wall_clock_time = int(sys.argv[2])
    outputdir = (file_path+
                 "/validation_output/GAST/run%d") % run_number

    start_validation(solver, instance_list, seed_list,
                     cutoff_time, wall_clock_time, outputdir)
