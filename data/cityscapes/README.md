### Preparation

1. Download all files from [CityPersonsOnRoad](https://github.com/eric612/CityPersonsOnRoad) at path $data

2. Download Images  from [cityscapes](https://www.cityscapes-dataset.com/) , and extract file into $data/JPEGImages

3. Download gtFine_trainvaltest.zip (241MB) [md5] from  [cityscapes](https://www.cityscapes-dataset.com/) , and extract files into $data/Annotations

4. Change datatset path in file "create_list_multi.sh" and "create_data.sh"

5. Create the LMDB file.

  ```Shell
  cd $CAFFE_ROOT
  # Create the val.txt, train.txt in data/cityscapes/
  bash data/cityscapes/create_list.sh
  # and make soft links at examples/cityscapes/
  bash data/cityscapes/create_data.sh
  ```

### Reference 

> https://github.com/dongdonghy/citypersons2voc