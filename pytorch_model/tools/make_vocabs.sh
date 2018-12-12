set -e
if [ $# -eq 0 ]; then
  echo "Need to pass in dataset like imagecaption or amazon or yelp"
else
  entire_corpus=data/$1/entire_corpus
  echo $entire_corpus
  rm -rf $entire_corpus
  cat data/$1/sentiment.train.0 >> $entire_corpus
  cat data/$1/sentiment.train.1 >> $entire_corpus
  cat data/$1/sentiment.dev.0 >> $entire_corpus
  cat data/$1/sentiment.dev.1 >> $entire_corpus
  cat data/$1/sentiment.test.0 >> $entire_corpus
  cat data/$1/sentiment.test.1 >> $entire_corpus
  echo "Making combined vocab"
  python tools/make_vocab.py $entire_corpus 16000 > data/$1/vocab.16k
  echo "Making attribute vocab"
  python tools/make_attribute_vocab.py data/$1/vocab.16k data/$1/sentiment.train.0 data/$1/sentiment.train.1 2  > data/$1/0.attribute_vocab.16k
  echo "Making attribute vocab 2"
  python tools/make_attribute_vocab.py data/$1/vocab.16k data/$1/sentiment.train.1 data/$1/sentiment.train.0 2  > data/$1/1.attribute_vocab.16k
  echo "Done"

fi