cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="/media/data/"
dataset_name="tusimple"
#mapfile="$root_dir/data/$dataset_name/labelmap.prototxt"
anno_type="lane_detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
label_type="json"
extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in train
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type  \
  --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --label-type=$label_type \
  --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt \
  $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
