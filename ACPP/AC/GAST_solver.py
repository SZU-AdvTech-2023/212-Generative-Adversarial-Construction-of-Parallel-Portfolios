#!/usr/bin/env python2.7
# change solver to full solver
import sys
import subprocess
import os

if __name__ == "__main__":
    params = ' '.join(sys.argv[6:])
    with open(os.path.abspath(os.path.realpath(__file__)+'/../exsiting_solver_Glo.txt'), 'r') as f:
        lines = f.readlines()
        existing_solver = ' ' + lines[0].strip() + ' '
        algNum = int(lines[1].strip())
    params = existing_solver + params.replace('-@1', '-@%d' % (algNum+1))
    sys.argv[1] = '"' + sys.argv[1] + '"'
    newparams = ' '.join(sys.argv[1:6]) + ' ' + params
    cmd = 'python '+os.path.abspath(os.path.realpath(__file__)+'/../../')+\
           ('/src/util/parallel_solver_wrapper.py %s') % newparams
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    print p.communicate()[0]
