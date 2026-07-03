#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define PI                3.1415926535897932384626422832795028841971f
#define INV_PI            0.31830988618379067154f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.0001f

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

class Aabb
{
public:
	__host__ __device__ Aabb() { reset(); }

	__device__ Aabb(const glm::vec3& p) : m_min(p), m_max(p) {}

	__host__ __device__ Aabb(const glm::vec3& mi, const glm::vec3& ma) : m_min(mi), m_max(ma) {}

	__host__ __device__ Aabb(const Aabb& rhs, const Aabb& lhs)
	{
		m_min = min(lhs.m_min, rhs.m_min);
		m_max = max(lhs.m_max, rhs.m_max);
	}

	__host__ __device__ Aabb(const Aabb& rhs) : m_min(rhs.m_min), m_max(rhs.m_max) {}

	__host__ __device__ void reset(void)
	{
		m_min = glm::vec3{ FLT_MAX, FLT_MAX, FLT_MAX };
		m_max = glm::vec3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	}

	__host__ __device__ Aabb& grow(const Aabb& rhs)
	{
		m_min = min(m_min, rhs.m_min);
		m_max = max(m_max, rhs.m_max);
		return *this;
	}

	__host__ __device__ Aabb& grow(const glm::vec3& p)
	{
		m_min = min(m_min, p);
		m_max = max(m_max, p);
		return *this;
	}

	__host__ __device__ glm::vec3 center() const { return (m_max + m_min) * 0.5f; }

	__host__ __device__ glm::vec3 extent() const { return m_max - m_min; }

	__host__ __device__ int maximumExtentDim() const {
		glm::vec3 d = extent();
		if (d.x > d.y && d.x > d.z)
			return 0;
		else if (d.y > d.z)
			return 1;
		else
			return 2;
	}

	__host__ __device__ float area() const
	{
		glm::vec3 ext = extent();
		return 2 * (ext.x * ext.y + ext.x * ext.z + ext.y * ext.z);
	}

	__host__ __device__ glm::vec3 offset(const glm::vec3& p) const
	{
		glm::vec3 o = p - m_min;
		if (m_max.x > m_min.x) o.x /= m_max.x - m_min.x;
		if (m_max.y > m_min.y) o.y /= m_max.y - m_min.y;
		if (m_max.z > m_min.z) o.z /= m_max.z - m_min.z;
		return o;
	}

	__host__ __device__ bool valid(void) { return m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z; }

	__host__ __device__ void intersect(const Aabb& box)
	{
		m_min = max(m_min, box.m_min);
		m_max = min(m_max, box.m_max);
	}

	__host__ __device__ glm::vec2 intersect(const glm::vec3& from, const glm::vec3& invRay, float maxt) const
	{
		const glm::vec3 dFar = (m_max - from) * (invRay);
		const glm::vec3 dNear = (m_min - from) * (invRay);
		const glm::vec3 tFar = max(dFar, dNear);
		const glm::vec3 tNear = min(dFar, dNear);
		float minFar = fmin(tFar.x, fmin(tFar.y, tFar.z));
		float maxNear = fmax(tNear.x, fmax(tNear.y, tNear.z));

		minFar = fmin(maxt, minFar);
		maxNear = fmax(0.0f, maxNear);

		return { maxNear, minFar };
	}

	glm::vec3 m_min;
	glm::vec3 m_max;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
