#ifndef CUPHOTO_LIB_EXCEPTION_HPP
#define CUPHOTO_LIB_EXCEPTION_HPP

#include <exception>
#include <string>

namespace cuphoto {

class CuPhotoException : public std::exception {
public:
    CuPhotoException(const std::string& msg) : message(msg) {}
    CuPhotoException(const char* msg) : message(msg) {}

    char* what() {
        return message.data();
    }

    friend std::ostream& operator<<(std::ostream& os, const CuPhotoException& crexc) { 
        return os << crexc.message;
    }
private:
    std::string message;
};

};

#endif