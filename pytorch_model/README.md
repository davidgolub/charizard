
This is an implementation of the DeleteOnly and DeleteAndRetrieve models from this paper: https://arxiv.org/pdf/1804.06437.pdf

# Usage

### Training

`python train.py --config sample_config.json --bleu`

This will train a model using the parameters in `sample_config.json`. Checkpoints, logs, decodings, and TensorBoard summaries will go into config's `working_dir`.


### Vocab generation

Given two corpus files, use the scripts in `tools/` to generate a vocabulary and attribute vocabulary:

```
python tools/make_vocab.py [entire corpus] [vocab size] > vocab.txt
python tools/make_attribute_vocab.py vocab.txt [corpus src file] [corpus tgt file] [salience ratio] > attribute_vocab.txt
```



# NOTE!!

This code does **NOT** implement step 2 of these algorithms (the "retrieve step", where given a src sentence you find a tgt sentence whose content is similar to the src according to TFIDF) because the data I'm working with is already aligned. We need to implement this step, but it should be pretty easy to do...this is generally how I think I would approach it:

1. Make a new version of the `data.WordDistance` object that uses a `TfidfVectorizer` instead of a `CountVectorizer` and get rid of the binary stuff
2. "align" your train data: use this tfidf distance class to either (a) preprocess your tgt corpus to be approximately aligned with the src corpus, or (b) re-arrange the tgt corpus after it's been read in `data.read_nmt_data()`. 
3. "align" your test data: either (a) Give the entire target corpus to `data.minibatch()` at inference time and select the matching tgt examples, or (b) during pre-processing, make a seperate "approximately aligned" tgt-side version of your src-side test data, where you selected examples from the tgt-side train data

I'm happy to do this, but it could also be a good way to familiarize yourself with the algorithm/code base? let me know what you want to do





# Algorithm

This is what this code implements. Slightly different than what's described in the paper (See section **NOTE!!**)


## Annotation

- x = all words in a sentence
- c(x) the _content_ words of a sentence, those that went unchanged
- m(x) the _modified_ words of a sentence, those that *were* changed
- v (= pre/post), the attribute of that sentence

## Delete
Train:
```
e = [ RNN( c(x_v) ), embedding_v ]
x_hat = RNN( e )
```

Test:
```
e = [ RNN( c(x_pre) ), embedding_post ]
x_post_hat = RNN(e)
```


## Delete AND Retrieve

Train
```
a'(x_v) = {
	 0.9: a(x_v)
	 0.1: another a(x'_v) within word-edit distance 1
}
e = [ RNN(c(x_v)), RNN(a'(x_v)) ]
x_v_hat = RNN(e)
```


Test
```
x'_post = argmin_{ x'_post (possibly excluding the true match) } ( tfidf( c(x'_post), c(x_pre) ) )
e = [ RNN( c(x_pre) ), RNN( a(x'_post) ) ]
x_post_hat = RNN(e)
```