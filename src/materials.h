#pragma once
#include "utilities.h"
#include <cuda_runtime.h>

enum MaterialType
{
    Lambertian,
    Specular,
};



struct Material
{
    __device__ void createMaterialInst(const Material& mat);

    __device__ glm::vec3 samplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 lambertianSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 specularSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ float getPDF(const glm::vec3& nor, glm::vec3 wo, glm::vec3 wi);

    __device__ glm::vec3 getBSDF(const glm::vec3& nor, glm::vec3 wo, glm::vec3 wi, float* pdf);

    glm::vec3 albedo = glm::vec3(0.5f);
    MaterialType type = MaterialType::Lambertian;
    float metallic = 0.f;
    float roughness = 1.f;
    float ior = 0.f;
    float emittance = 0.f;
};