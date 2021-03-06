main_operation=$1
main_function=$2
main_data=$3
main_dict_num=$4
main_dict_thre=$5
main_dev_num=$6

if [ "$main_data" = "amazon" ]; then
  batch_size=64
else
  batch_size=256
fi

main_function_orgin=$main_function
if [ "$main_function" = "DeleteOnly" ]; then
main_function=label
elif [ "$main_function" = "DeleteAndRetrieve" ]; then
main_function=orgin
elif [ "$main_function" = "RetrieveOnly" ]; then
main_function=orgin
fi

if [ ! -n "$main_dict_num" ]; then
main_dict_num=7000
main_dict_thre=15
main_dev_num=4000
elif [ "$main_data" = "imagecaption" ]; then
if [ ! -n "$main_dict_num" ]; then
main_dict_num=3000
main_dict_thre=5
fi
main_dev_num=1000
elif [ "$main_data" = "republican" ]; then
if [ ! -n "$main_dict_num" ]; then
main_dict_num=3000
main_dict_thre=5
fi
main_dev_num=1000
elif [ "$main_data" = "amazon" ]; then
if [ ! -n "$main_dict_num" ]; then
main_dict_num=10000
main_dict_thre=5.5
fi
main_dev_num=2000
fi

main_category=sentiment
main_category_num=2
#configure
preprocess_tool_path=src/tool/
data_path=data/
data_dir=${data_path}${main_data}/
train_file_prefix=${data_path}${main_data}/${main_category}.train.
dev_file_prefix=${data_path}${main_data}/${main_category}.dev.
test_file_prefix=${data_path}${main_data}/${main_category}.test.
orgin_train_file_prefix=$train_file_prefix
orgin_dev_file_prefix=$dev_file_prefix
orgin_test_file_prefix=$test_file_prefix
train_data_file=${data_dir}/train.data.${main_function}
test_data_file=${data_dir}/test.data.${main_function}
dict_data_file=${data_dir}/zhi.dict.${main_function}

if [ "$main_operation" = "preprocess" ]; then
# python2.7 ${preprocess_tool_path}filter_style_ngrams.py $orgin_train_file_prefix $main_category_num $main_function $train_file_prefix
python2.7 ${preprocess_tool_path}filter_style_ngrams.py $orgin_train_file_prefix $main_category_num $main_function $orgin_train_file_prefix
# if [ "$main_data" = "amazon" ]; then
# echo "done preprocessing"
# for((i=0;i < $main_category_num; i++))
# do
#         python2.7 ${preprocess_tool_path}use_nltk_to_filter.py ${train_file_prefix}${i}.tf_idf.$main_function
#         cp ${train_file_prefix}${i}.tf_idf.${main_function}.filter ${train_file_prefix}${i}.tf_idf.$main_function
# done
# echo "done preprocessing yet again"
# fi

printf "\n\n\npreprocessing data!!!!!!!!!!!!\n"
for((i=0;i < $main_category_num; i++))
do
    echo "Processing on category " $i
    python2.7 ${preprocess_tool_path}preprocess_train.py ${orgin_train_file_prefix}${i} ${orgin_train_file_prefix}${i} ${main_function} ${main_dict_num} ${main_dict_thre} ${train_file_prefix}${i}
    python2.7 ${preprocess_tool_path}preprocess_test.py ${orgin_dev_file_prefix}${i} ${orgin_train_file_prefix}${i} $main_function $main_dict_num $main_dict_thre ${dev_file_prefix}${i}
done

echo "copying files"
cat ${train_file_prefix}*.data.${main_function} >> $train_data_file
cat ${dev_file_prefix}*.data.${main_function} >> $test_data_file

echo "shuffling"
python2.7 ${preprocess_tool_path}shuffle.py $train_data_file
python2.7 ${preprocess_tool_path}shuffle.py $test_data_file

echo "creating dict"
cat ${test_data_file}.shuffle >>${train_data_file}.shuffle
cp ${train_data_file}.shuffle ${train_data_file}
python2.7 ${preprocess_tool_path}create_dict.py ${train_data_file} $dict_data_file

elif [ "$main_operation" = "train" ]; then

line_num=$(wc -l < $train_data_file)
vt=$main_dev_num

eval $(awk 'BEGIN{printf "train_num=%.6f",'$line_num'-'$vt'}')
test_num=$main_dev_num
vaild_num=0
eval $(awk 'BEGIN{printf "train_rate=%.6f",'$train_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "vaild_rate=%.6f",'$vaild_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "test_rate=%.6f",'$test_num'/'$line_num'}')
echo "Now we're TRAINING!!!!"
#train process
THEANO_FLAGS="${THEANO_FLAGS}" python2.7 src/main.py ../model $train_data_file $dict_data_file src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT train $batch_size

elif [ "$main_operation" = "test" ]; then
echo test
printf "\n\n\npreparing data\n"
if [ "$main_function" = "TemplateBased" ]; then
python2.7 ${preprocess_tool_path}filter_style_ngrams.py $orgin_train_file_prefix $main_category_num $main_function $train_file_prefix
for((i=0;i < $main_category_num; i++))
do
	python2.7 ${preprocess_tool_path}prepare_templatebased_data.py ${train_file_prefix}$i ${orgin_train_file_prefix}$i ${train_file_prefix}$i $main_dict_thre $main_dict_num
	python2.7 ${preprocess_tool_path}prepare_templatebased_data.py ${train_file_prefix}$i ${orgin_test_file_prefix}$i ${test_file_prefix}$i $main_dict_thre $main_dict_num
done
mkdir sen1
mkdir sen0
python2.7 ${preprocess_tool_path}retrieve_mode_my_1.py
python2.7 ${preprocess_tool_path}retrieve_mode_my_0.py
for((i=0;i < $main_category_num; i++))
do
	python2.7 ${preprocess_tool_path}build_templatebased.py ${test_file_prefix}${i}.template1.result ${orgin_test_file_prefix}${i}
	cp ${test_file_prefix}${i}.template1.result ${test_file_prefix}${i}.${main_function}.$main_data
done

exit
fi


#preprocess test data
line_num=$(wc -l < $train_data_file)
vt=$main_dev_num
eval $(awk 'BEGIN{printf "train_num=%.6f",'$line_num'-'$vt'}')
test_num=$main_dev_num
vaild_num=0
eval $(awk 'BEGIN{printf "train_rate=%.6f",'$train_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "vaild_rate=%.6f",'$vaild_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "test_rate=%.6f",'$test_num'/'$line_num'}')
printf "preprocessing test data\n"
for((i=0;i<$main_category_num;i++))
do
        python2.7 ${preprocess_tool_path}preprocess_test.py ${orgin_test_file_prefix}${i} ${train_file_prefix}${i} $main_function $main_dict_num $main_dict_thre ${test_file_prefix}${i}
        python2.7 ${preprocess_tool_path}filter_template_test.py ${test_file_prefix}${i} ${main_function}
        python2.7 ${preprocess_tool_path}filter_template.py ${train_file_prefix}${i} ${main_function}
done

printf "\n\n\ngenerate embeddings\n"
for((i=0;i<$main_category_num;i++))
do
	THEANO_FLAGS="${THEANO_FLAGS}" python2.7 src/main.py ../model $train_data_file $dict_data_file src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT generate_emb ${test_file_prefix}${i}.template.${main_function} $batch_size
  THEANO_FLAGS="${THEANO_FLAGS}" python2.7 src/main.py ../model $train_data_file $dict_data_file src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT generate_emb ${train_file_prefix}${i}.template.${main_function} $batch_size
done

printf "\n\n\npreprocessing more test data!!!!!\n"
cd src/tool
echo $main_operation $main_function $main_data $main_category $main_category_num $main_dict_num $main_dict_thre
bash preprocess_test_data.sh $main_operation $main_function $main_data $main_category $main_category_num $main_dict_num $main_dict_thre
cd ../..

printf "\n\n\nfind nearst neighbot and form test data\n"
for((i=0;i<$main_category_num;i++))
do
	python2.7 ${preprocess_tool_path}find_nearst_neighbot_all.py $i $main_data ${main_function} ${data_dir}
	python2.7 ${preprocess_tool_path}form_test_data.py ${test_file_prefix}${i}.template.${main_function}.emb.result
done


#test process
printf "\n\n\ntest process\n"
for((i=0;i<$main_category_num;i++))
do 
THEANO_FLAGS="${THEANO_FLAGS}" python2.7 src/main.py ../model $train_data_file $dict_data_file src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT generate_b_v_t ${test_file_prefix}${i}.template.${main_function}.emb.result.filter $batch_size
done
if [ "$main_function_orgin" = "RetrieveOnly" ]; then
python2.7 ${preprocess_tool_path}get_retrieval_result.py ${data_dir}
for((i=0;i<$main_category_num;i++))
do
cp ${test_file_prefix}${i}.retrieval ${test_file_prefix}${i}.${main_function_orgin}.$main_data
done
exit
fi

echo "building data, shuffling, creating dicts"
for((i=0;i<$main_category_num;i++))
do
	python2.7 ${preprocess_tool_path}build_lm_data.py ${orgin_train_file_prefix}${i} ${train_file_prefix}${i}
	python2.7 ${preprocess_tool_path}shuffle.py ${train_file_prefix}${i}.lm
	cp ${train_file_prefix}${i}.lm.shuffle ${train_file_prefix}${i}.lm
	python2.7 ${preprocess_tool_path}create_dict.py ${train_file_prefix}${i}.lm ${train_file_prefix}${i}.lm.dict
done


printf "\n\n\n ChoEncoderDecoderLm\n"
for((i=0;i<$main_category_num;i++))
do
	vaild_num=$i
	eval $(awk 'BEGIN{printf "vaild_rate=%.10f",'$vaild_num'/'$line_num'}')
	THEANO_FLAGS="${THEANO_FLAGS}" python2.7 src/main.py ../model ${train_file_prefix}${i}.lm ${train_file_prefix}${i}.lm.dict src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderLm train $batch_size
done
vaild_num=0

printf "\n\n\n ChoEncoderDecoderLm generate_b_v_t_v\n"
i=0
eval $(awk 'BEGIN{printf "vaild_rate=%.10f",'$vaild_num'/'$line_num'}')
THEANO_FLAGS="${THEANO_FLAGS}" python2.7 src/main.py ../model ${train_file_prefix}${i}.lm ${train_file_prefix}${i}.lm.dict src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderLm generate_b_v_t_v ${test_file_prefix}1.template.${main_function}.emb.result.filter.result $batch_size
vaild_num=1
i=1
eval $(awk 'BEGIN{printf "vaild_rate=%.10f",'$vaild_num'/'$line_num'}')
THEANO_FLAGS="${THEANO_FLAGS}" python2.7 src/main.py ../model ${train_file_prefix}${i}.lm ${train_file_prefix}${i}.lm.dict src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderLm generate_b_v_t_v ${test_file_prefix}0.template.${main_function}.emb.result.filter.result $batch_size

for((i=0;i<$main_category_num;i++))
do
  python2.7 ${preprocess_tool_path}get_final_result.py ${test_file_prefix}${i}.template.${main_function}.emb.result.filter.result ${i}
	cp ${test_file_prefix}${i}.template.${main_function}.emb.result.filter.result.final_result ${test_file_prefix}${i}.${main_function_orgin}.$main_data

done

fi
