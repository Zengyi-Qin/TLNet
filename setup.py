import os

# download the pretrained model
if not os.path.exists('./avod/models'):
    os.system('wget https://cloud.tsinghua.edu.cn/f/6eb6b856efec42d19937/?dl=1 -O tlnet_pretrained.tar.gz')
    os.system('tar -xf tlnet_pretrained.tar.gz')
    os.system('rm tlnet_pretrained.tar.gz')

# compile protobuf
os.system('sh avod/protos/run_protoc.sh')
# generate minibatches
os.system('python scripts/preprocessing/gen_mini_batches.py')
