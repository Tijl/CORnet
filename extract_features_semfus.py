import os,sys,subprocess,glob

fun = 'extract_features.py'
model = 'S'
sublayer = 'output'
data_path = '~/Desktop/CORNET-input'
output_path = '~/Desktop/CORNET-output'

for layer in ['V1', 'V2', 'V4', 'IT', 'decoder']:
    cmd = 'python %s test --model %s --layer %s --sublayer %s --data_path %s --output_path %s'%(
        fun,model,layer,sublayer,data_path,output_path)
    print(subprocess.call(cmd, shell=True))
