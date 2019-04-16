cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="/media/data/"
dataset_name="bdd100k"
mapfile="$root_dir/data/$dataset_name/labelmap_fisheye.prototxt"
anno_type="detection_with_segmentation"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
#for subset in test test_time_Afternoon test_time_Day test_time_Night test_weather_Cloudy test_weather_Normal test_weather_Rain trainval
for subset in train
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type \
  --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width \
  --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt \
  $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
