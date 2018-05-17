# - Try to find ALE
#
# The following variables are optionally searched for defaults
#  ALE_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done:
#  ALE_FOUND
#  ALE_INCLUDE_DIRS
#  ALE_LIBRARIES

include(FindPackageHandleStandardArgs)

set(ALE_ROOT_DIR "" CACHE PATH "/home/pfzhu/Documents/ALE-0.5.1")

find_path(ALE_INCLUDE_DIR ale_interface.hpp
  PATHS ${ALE_ROOT_DIR}
  PATH_SUFFIXES
  src)

find_library(ALE_LIBRARY ale
  PATHS ${ALE_ROOT_DIR}
  PATH_SUFFIXES
  src)

find_package_handle_standard_args(ALE DEFAULT_MSG
  ALE_INCLUDE_DIR ALE_LIBRARY)

if(ALE_FOUND)
  set(ALE_INCLUDE_DIRS ${ALE_INCLUDE_DIR})
  set(ALE_LIBRARIES ${ALE_LIBRARY})
endif()