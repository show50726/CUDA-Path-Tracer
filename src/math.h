#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"
#include "utilities.h"

namespace math {

__device__ inline float absDot(const glm::vec3& a, const glm::vec3& b) {
    return glm::abs(glm::dot(a, b)); 
}

__device__ inline float cosTheta(const glm::vec3& w) {
    return w.z;
}

__device__ inline float absCosTheta(const glm::vec3& w) {
    return glm::abs(w.z); 
}

__device__ inline glm::vec2 sampleUniformDisk(const glm::vec2& u) {
    float theta = u.y * TWO_PI;
    return glm::vec2(glm::cos(theta), glm::sin(theta)) * glm::sqrt(u.x);
}

__device__ inline glm::vec3 sampleHemisphereCosine(const glm::vec2& u) {
    glm::vec2 d = sampleUniformDisk(u);
    return glm::vec3(d, glm::sqrt(1.0f - glm::dot(d, d)));
}

__device__ inline float cosineHemispherePDF(float cosTheta) {
    return cosTheta * INV_PI;
}

__device__ inline glm::vec3 faceforward(const glm::vec3& n, const glm::vec3& v) {
    return (glm::dot(n, v) < 0.f) ? -n : n;
}

// Reference: pbrt-v3 reflection.cpp
__device__ inline float frDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cosThetaI = glm::abs(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    float sinThetaI = glm::sqrt(glm::max(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1) return 1;
    float cosThetaT = glm::sqrt(glm::max(0.0f, 1.0f - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
}

} // namespace math