import os
import shutil

shrink_dirs = ['yelp', 'amazon']
shrink_factor = 3

for shrink_dir in shrink_dirs:
  new_dir = '{}_mini/'.format(shrink_dir)
  if not os.path.isdir(new_dir):
    os.makedirs(new_dir)

  for i in range(2):
    filename = '{}/sentiment.train.{}'.format(shrink_dir, i)
    # print(filename)
    with open(filename,'r') as f:
      lines = f.readlines()
      orig_len = len(lines)
      lines = lines[:len(lines) // shrink_factor]
      print("new vs old lengths: {} vs {}".format(len(lines), orig_len))
    new_filename = '{}/sentiment.train.{}'.format(new_dir, i)
    print(new_filename)
    with open(new_filename,'w') as f:
      for line in lines:
        f.write(line)   


  for split in ['test','dev']:
    for i in range(2):
      shutil.copyfile('{}/sentiment.{}.{}'.format(shrink_dir, split, i), '{}/sentiment.{}.{}'.format(new_dir, split, i))