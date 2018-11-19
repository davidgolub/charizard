mv "preprocessed_data/askmen + askwomen" model/data/askmenwomen
THEANO_FLAGS='device=cuda0,floatX=float32' bash run.sh train DeleteAndRetrieve askmenwomen
THEANO_FLAGS='device=cuda0,floatX=float32' bash run.sh test DeleteAndRetrieve askmenwomen
bash run.sh train DeleteAndRetrieve yelp
bash run.sh train DeleteAndRetrieve yelp



