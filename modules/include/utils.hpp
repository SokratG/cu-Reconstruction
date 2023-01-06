#ifndef CUREC_LIB_UTILS_HPP
#define CUREC_LIB_UTILS_HPP

#include <unistd.h>
#include <chrono>

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/functional/hash.hpp>

#include <glog/logging.h>

namespace curec {

class TimeProfile
{
public:
    TimeProfile(bool show = false) {
        start = std::chrono::high_resolution_clock::now();
        showTimeDestr = show;
    }
    ~TimeProfile() {
        end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (showTimeDestr)
            LOG(INFO) << duration.count() << " ms\n"; //µs
    }
    int64_t getTime() noexcept
    {
        end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    bool showTimeDestr = false;
};


class UUID 
{
public:
    static boost::uuids::random_generator gen;
};

};

namespace std
{
// std::unordered_map<boost::uuids::uuid, T, boost::hash<boost::uuids::uuid>>
template<>
struct hash<boost::uuids::uuid>
{
    size_t operator()(const boost::uuids::uuid& uid)
    {
        return boost::hash<boost::uuids::uuid>()(uid);
    }
};
}

#endif