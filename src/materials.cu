#include "materials.h"

#include "math.h"

// ref: http://marc-b-reynolds.github.io/quaternions/2016/07/06/Orthonormal.html
__device__ static glm::mat3 getTBN(const glm::vec3& N)
{
	float x = N.x, y = N.y, z = N.z;
	float sz = z < 0.f ? -1.f : 1.f;
	float a = y / (z + sz);
	float b = y * a;
	float c = x * a;
	glm::vec3 T = glm::vec3(-z - b, c, x);
	glm::vec3 B = glm::vec3(sz * c, sz * b - 1, sz * y);

	return glm::mat3(T, B, N);
}

__device__ void Material::createMaterialInst(const Material& mat)
{
	albedo = mat.albedo;
	roughness = mat.roughness;
	metallic = mat.metallic;
}

__device__ glm::vec3 Material::samplef(const glm::vec3& nor, 
	glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
	switch (type)
	{
	case Lambertian:
		return lambertianSamplef(nor, wo, wi, rng, pdf);
		break;
	case Specular:
		return specularSamplef(nor, wo, wi, rng, pdf);
		break;
	}
}

__device__ glm::vec3 Material::lambertianSamplef(const glm::vec3& nor, 
	glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf) {

	wi = math::sampleHemisphereCosine(glm::vec2(rng.x, rng.y));
	if (wi.z < 0)
		wi.z *= -1.0f;
	
	*pdf = math::cosineHemispherePDF(math::absCosTheta(wi));

	// to world space
	glm::mat3 TBN = getTBN(nor);
	wi = TBN * wi;
	return albedo * INV_PI;
}

__device__ glm::vec3 Material::specularSamplef(const glm::vec3& nor, 
	glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
	*pdf = 1.f;
	wi = glm::reflect(wo, nor);
	return glm::vec3(1.f);
}