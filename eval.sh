mv "preprocessed_data/Republican + Democrat" model/data/republican
THEANO_FLAGS='device=cuda0,floatX=float32,base_compiledir=/mnt/u/golubd' bash run.sh train DeleteAndRetrieve republican
THEANO_FLAGS='device=cuda0,floatX=float32,base_compiledir=/mnt/u/golubd' bash run.sh test DeleteAndRetrieve republican
bash run.sh train DeleteAndRetrieve yelp
bash run.sh train DeleteAndRetrieve yelp



