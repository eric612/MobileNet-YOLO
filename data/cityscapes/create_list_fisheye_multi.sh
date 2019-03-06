#!/bin/bash
root_dir=/media/data/
#root_dir="../../data/VOCdevkit"
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in train val
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in fisheye_workspace #VOC2007 VOC2012
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.png/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file
    
    label_file2=$bash_dir/$dataset"_label2.txt"
    cp $dataset_file $label_file2
    sed -i "s/^/$name\/Annotations\//g" $label_file2
    sed -i "s/$/_gtFine_labelIds.png/g" $label_file2
    
    paste -d' ' $img_file $label_file $label_file2>> $dst_file

    rm -f $label_file
    rm -f $label_file2
    rm -f $img_file
  done
  # Shuffle trainval file.
  if [ $dataset == "train" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
