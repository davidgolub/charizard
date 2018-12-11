python tools/make_vocab.py test_data/$1/sentiment.train.0 16000 > test_data/$1/train.0.vocab.16k
python tools/make_vocab.py test_data/$1/sentiment.train.1 16000 > test_data/$1/train.1.vocab.16k
python tools/make_attribute_vocab.py test_data/$1/train.0.vocab.16k test_data/$1/sentiment.train.0 test_data/$1/sentiment.train.1 2  > test_data/$1/0.attribute_vocab.16k
python tools/make_attribute_vocab.py test_data/$1/train.1.vocab.16k test_data/$1/sentiment.train.1 test_data/$1/sentiment.train.0 2  > test_data/$1/1.attribute_vocab.16k
