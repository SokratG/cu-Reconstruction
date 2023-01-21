#include "feature.hpp"

namespace cuphoto {

bool Feature::outlier() const {
    return is_outlier;
}

void Feature::outlier(const bool v) {
    is_outlier = v;
}

};