main_operation=$1
main_function=$2
main_data=$3
main_category=$4
main_category_num=$5
main_dict_num=$6
main_dict_thre=$7

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


if [ "$main_data" = "RepublicanDemocrat" ]; then
if [ ! -n "$main_dict_num" ]; then
main_dict_num=7000
main_dict_thre=15
fi
main_dev_num=4000
elif [ "$main_data" = "imagecaption" ]; then
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


# train_file_prefix=../${main_data}/${main_category}/data.train.
# test_file_prefix=../${main_data}/${main_category}/data.test.
train_file_prefix=../../data/${main_data}/${main_category}.train.
test_file_prefix=../../data/${main_data}/${main_category}.test.

for((i=0;i<$main_category_num;i++))
do
	python preprocess_test.py ${test_file_prefix}${i} ${train_file_prefix}${i} $main_function $main_dict_num $main_dict_thre ${test_file_prefix}${i}
	python filter_template_test.py ${test_file_prefix}${i} ${main_function}
	python filter_template.py ${train_file_prefix}${i} ${main_function}
done
<<BLOCK
python prepare_test_data.py sentiment.test.0 sentiment.train.0
python prepare_test_data.py sentiment.test.1 sentiment.train.1
cp sentiment.train.1.data.test sentiment.test.1.data
cp sentiment.train.0.data.test sentiment.test.0.data
python filter_template_test.py sentiment.test.1
python filter_template_test.py sentiment.test.0
python filter_template.py sentiment.train.1
python filter_template.py sentiment.train.0
cp *.template /data1/qspace/juncenli/template_style_transform/data/style_transfer/noise_slot_data/
BLOCK
