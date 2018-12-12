saliency_threshold=5


set -e
entire_corpus=data/transfer/entire_corpus
positive_corpus=data/transfer/positive_corpus
negative_corpus=data/transfer/negative_corpus
echo $entire_corpus
rm -rf $entire_corpus
rm -rf $positive_corpus
rm -rf $negative_corpus
# Combine corpuses
cat data/yelp_shrunken/sentiment.train.0 >> $entire_corpus
cat data/yelp_shrunken/sentiment.train.1 >> $entire_corpus
cat data/amazon_shrunken/sentiment.train.0 >> $entire_corpus
cat data/amazon_shrunken/sentiment.train.1 >> $entire_corpus

cat data/yelp_shrunken/sentiment.train.1 >> $positive_corpus
cat data/amazon_shrunken/sentiment.train.1 >> $positive_corpus

cat data/yelp_shrunken/sentiment.train.0 >> $negative_corpus
cat data/amazon_shrunken/sentiment.train.0 >> $negative_corpus


echo "Making combined vocab"
python tools/make_vocab.py $entire_corpus 16000 > data/transfer/vocab.16k
echo "Making attribute vocab"
python tools/make_attribute_vocab.py data/transfer/vocab.16k $negative_corpus $positive_corpus $saliency_threshold  > data/transfer/0.attribute_vocab.16k
echo "Making attribute vocab 2"
python tools/make_attribute_vocab.py data/transfer/vocab.16k $positive_corpus $negative_corpus $saliency_threshold  > data/transfer/1.attribute_vocab.16k
echo "Done"
