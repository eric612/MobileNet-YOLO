### Preparation

1. Download Images  from [cityscapes](https://www.cityscapes-dataset.com/) , and download annomation from [CityPersonsOnRoad](https://github.com/eric612/CityPersonsOnRoad)

2. Change datatset path in file "create_list.sh" and "create_data.sh"

3. Create the LMDB file.

  ```Shell
  cd $CAFFE_ROOT
  # Create the val.txt, train.txt in data/cityscapes/
  bash data/cityscapes/create_list.sh
  # and make soft links at examples/cityscapes/
  bash data/cityscapes/create_data.sh
  ```

### Reference 

> https://github.com/dongdonghy/citypersons2voc