import os
import random
# files = '/home/liuwei/RIPC/SAT0/instances'
# # output_file = "/home/liuwei/RIPC/Core8_0/trainIndices.txt"
# fileList = os.listdir(files)
# featureFile = '/home/liuwei/RIPC/SAT0/indices/whole_instance_feature'
# testFile = '/home/liuwei/RIPC/SAT0/indices/test'
# trainFile = '/home/liuwei/RIPC/SAT0/indices/training'
# soluFile = '/home/liuwei/GAST/ACPP/instance_set/SAT2/indices/solubility.txt'
# allFile=[]
# for i in fileList:
#     allFile.append(i)
# with open(output_file, 'w+') as ff:
#     for f in range(1,601):
#             ff.write(files+'/%d.tsp\n'%f)
    # oldDir = os.path.join(files, f)
    # filetype = os.path.splitext(f)[1]
    # newDir = os.path.join(files, str(run) + filetype)
    # os.rename(oldDir, newDir)
    # with open(featureFile, "r") as ff:
    #     lines = ff.readlines()
    # with open(testFile, "r") as ff:
    #     test = ff.readlines()
    # with open(trainFile, "r") as ff:
    #     train = ff.readlines()
    # with open(trainFile, "r") as ff:
    #     solution = ff.readlines()
    # with open(featureFile, 'w') as ff:
    #     for line in lines:
    #         if f in line:
    #             line = line.replace(f, str(run) + filetype)
    #         ff.write(line)
    # with open(testFile, 'w') as tf:
    #     for t in test:
    #         if f in t:
    #             t = t.replace(f, str(run) + filetype)
    #         tf.write(t)
    # with open(trainFile, 'w') as Tf:
    #     for T in train:
    #         if f in T:
    #             T = T.replace(f, str(run) + filetype)
    #         Tf.write(T)
    # with open(soluFile, 'w') as Sf:
    #     for S in solution:
    #         if f in S:
    #             S = S.replace(f, str(run) + filetype)
    #         Sf.write(S)
# file1 = "/home/liuwei/GAST/ACPP/instance_set/SAT0/instances"
# file2 = "/home/liuwei/GAST/ACPP/instance_set/SAT2/instances"
# testfile = '/home/liuwei/GAST/ACPP/instance_set/SAT1/indices/test'
# trainfile = '/home/liuwei/GAST/ACPP/instance_set/SAT1/indices/training'
# filelist1 = os.listdir(file1)
# filelist2 = os.listdir(file2)
# allFile = []
# for i in filelist1:
#     allFile.append(i)
# for i in filelist2:
#     allFile.append(i)
# with open(trainfile,'w+') as f:
#     for i in range(1,601):
#         ran=random.randint(0,len(allFile)-1)
#         if  allFile[ran].__contains__('cbmc'):
#             f.write(file1 + '/%s\n' % allFile[ran])
#         else:
#             f.write(file2 + '/%s\n' % allFile[ran])
#         allFile.remove(allFile[ran])
# with open(testfile,'w+') as f:
#     for i in range(1,889):
#         ran=random.randint(0,len(allFile)-1)
#         if allFile[ran].__contains__('cbmc'):
#             f.write(file1 + '/%s\n' % allFile[ran])
#         else:
#             f.write(file2 + '/%s\n' % allFile[ran])
#         allFile.remove(allFile[ran])
# filedir='/home/liuwei/GAST/ACPP/Solver/paramfile/test.txt'
# filedir1='/home/liuwei/GAST/ACPP/Solver/paramfile/Mul_lingeling_ala_pcs4.txt'
#
# tem=[]
# with open(filedir,'r') as f:
#     for i in f:
#         tem.append(i)
# with open(filedir1, 'w+') as f:
#     for i in range(0,len(tem)):
#         c=tem[i].replace('@1','@8')
#         f.write(c)
file1 = "/home/liuwei/RIPC/SAT0/instances"
file2 = "/home/liuwei/GAST/ACPP/instance_set/SAT2/instances"
testfile = '/home/liuwei/RIPC/SAT0/indices/test'
trainfile = '/home/liuwei/RIPC/SAT0/indices/training'
filelist1 = os.listdir(file1)
filelist2 = os.listdir(file2)
allFile = []
for i in filelist1:
    print 100
    allFile.append(i)
    print allFile[i]
for file in filelist1:
    print file
    exit(0)