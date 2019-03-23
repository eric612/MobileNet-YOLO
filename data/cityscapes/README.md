### Demo Video 

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/SE_0MeN2nTw/0.jpg)](https://www.youtube.com/watch?v=SE_0MeN2nTw)

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
### Demo with pre-trained model 

1. Download demo images  from [cityscapes](https://www.cityscapes-dataset.com/) , and extract file into $data/images

2. Download [caffemodel](https://drive.google.com/open?id=1KghVSgH1FTsTOelHQdc-Y804RhjZSnWw) and save at models/cityscapes/

3. 
  ```Shell
  cd $CAFFE_ROOT
  sh demo_cityscapes.sh
  ```
  
### Reference 

> https://github.com/dongdonghy/citypersons2voc
