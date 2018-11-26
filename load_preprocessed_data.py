import os
from zipfile import ZipFile
from glob import glob
from shutil import rmtree, copytree

input_file =  'preprocessed_data.zip'
interm_dir = 'preprocessed_data/'
output_dir_prefix = 'model/data/'
# if os.path.isdir(interm_dir):
#   rmtree(interm_dir)

with ZipFile(input_file,'r') as zip:
  zip.extractall(interm_dir)

for dirname in glob(interm_dir + '/*'):
  for filename in glob(dirname + '/*'):
    new_filename = filename.replace('/data.', '/sentiment.')
    print(new_filename)
    os.rename(filename, new_filename)
  output_dir = output_dir_prefix + dirname.split('/')[-1]
  print(output_dir)
  copytree(dirname, output_dir)

# rmtree(interm_dir)