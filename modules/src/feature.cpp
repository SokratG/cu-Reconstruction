#include "feature.hpp"

namespace curec {

bool Feature::outlier() const {
    return is_outlier;
}

void Feature::outlier(const bool v) {
    is_outlier = v;
}

};