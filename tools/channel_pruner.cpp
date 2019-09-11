#include "caffe/Pruner.h"
#include<string>
int main(){
	std::string xml_path = "sys_test_config.xml";
	Pruner t = Pruner(xml_path);
	t.start();
	/*int i = 7 / 2;
	std::cout << i;*/
	//system("pause");
  return 1;
}




