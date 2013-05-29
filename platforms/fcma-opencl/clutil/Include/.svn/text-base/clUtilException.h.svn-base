#pragma once
#include "clUtilCommon.h"

#define clUtilCheckError(err)\
{\
  if(err != CL_SUCCESS)\
  {\
    std::stringstream stream(std::stringstream::in | std::stringstream::out);\
    \
    stream << (const char*)"Error "\
           << __FILE__\
           << ":"\
           << __LINE__\
           << ":"\
           << clUtilGetErrorCode(err);\
    \
    throw clUtilException(stream.str());\
  }\
}

//Turns error code into string
const char* clUtilGetErrorCode(cl_int err);

class clUtilException : std::exception
{
  public:
    clUtilException(std::string what) :
      mWhat(what)
    {
    }

    virtual const char* what() const throw()
    {
      return mWhat.c_str();
    }

    virtual ~clUtilException() throw()
    {
    }

  private:
    clUtilException();
    std::string mWhat;
};
