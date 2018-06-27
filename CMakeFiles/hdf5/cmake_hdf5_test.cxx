#include <H5Cpp.h>
#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif
const char* info_ver = "INFO" ":" H5_VERSION;
#ifdef H5_HAVE_PARALLEL
const char* info_parallel = "INFO" ":" "PARALLEL";
#endif
int main(int argc, char **argv) {
  int require = 0;
  require += info_ver[argc];
#ifdef H5_HAVE_PARALLEL
  require += info_parallel[argc];
#endif
  H5File file("foo.h5", H5F_ACC_TRUNC);
  return 0;
}