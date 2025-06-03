#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"
#include "utilities.h"

namespace math {

__device__ inline float absDot(const glm::vec3& a, const glm::vec3& b) {
    return glm::abs(glm::dot(a, b)); 
}

__device__ inline float absCosTheta(glm::vec3 w) {
    return glm::abs(w.z); 
}

__device__ inline glm::vec2 sampleUniformDisk(glm::vec2 u) {
    float theta = u.y * TWO_PI;
    return glm::vec2(glm::cos(theta), glm::sin(theta)) * glm::sqrt(u.x);
}

__device__ inline glm::vec3 sampleHemisphereCosine(glm::vec2 u) {
    glm::vec2 d = sampleUniformDisk(u);
    return glm::vec3(d, glm::sqrt(1.0f - glm::dot(d, d)));
}

__device__ inline float cosineHemispherePDF(float cosTheta) {
    return cosTheta * INV_PI;
}

} // namespace math