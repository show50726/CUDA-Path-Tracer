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
	case FresnelSpecular:
		return fresnelSamplef(nor, wo, wi, rng, pdf);
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
	wi = glm::reflect(wo, nor);
	*pdf = 1.f;
	return glm::vec3(1.f);
}

__device__ glm::vec3 Material::fresnelSamplef(const glm::vec3& nor,
	glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
	constexpr float ior_air = 1.0f;
	float cosTheta = math::cosTheta(wo);

	float F = math::frDielectric(cosTheta, ior_air, ior);
	if (rng[0] < F) {
		// Compute specular reflection for _FresnelSpecular_

		// Compute perfect specular reflection direction
		wi = glm::vec3(-wo.x, -wo.y, wo.z);
		*pdf = F;
		return F * reflectance / math::absCosTheta(wi);
	}
	else {
		// Compute specular transmission for _FresnelSpecular_

		// Figure out which $\eta$ is incident and which is transmitted
		bool entering = cosTheta > 0;
		float etaI = entering ? ior_air : ior;
		float etaT = entering ? ior : ior_air;

		// Compute ray direction for specular transmission
		wi = glm::refract(wi, math::faceforward(glm::vec3(0, 0, 1), wo), etaI / etaT);
		if (glm::all(glm::equal(wi, glm::vec3(0)))) {
			return glm::vec3(0.0f);
		}

		glm::vec3 ft = transmittance * (1 - F);

		*pdf = 1 - F;
		return ft / math::absCosTheta(wi);
	}

	wi = glm::reflect(wo, nor);
	*pdf = 1.f;
	return glm::vec3(1.f);
}