#ifndef CUREC_LIB_EXCEPTION_HPP
#define CUREC_LIB_EXCEPTION_HPP

#include <exception>
#include <string>

namespace curec {

class CuRecException : public std::exception {
public:
    CuRecException(const std::string& msg) : message(msg) {}
    CuRecException(const char* msg) : message(msg) {}

    char* what() {
        return message.data();
    }

    friend std::ostream& operator<<(std::ostream& os, const CuRecException& crexc) { 
        return os << crexc.message;
    }
private:
    std::string message;
};

};

#endif