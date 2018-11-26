# file for david liang to play around with and not have to type args all the time

set -e

# main_category=ps3_xbox
# main_category=republican_democrat
main_category=amazon
bash run.sh preprocess DeleteAndRetrieve ${main_category}
THEANO_FLAGS='device=cuda0,floatX=float32' bash run.sh train DeleteAndRetrieve ${main_category}
THEANO_FLAGS='device=cuda0,floatX=float32' bash run.sh test DeleteAndRetrieve ${main_category}