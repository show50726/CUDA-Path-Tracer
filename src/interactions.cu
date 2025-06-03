#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

//__host__ __device__ void scatterRay(
//    PathSegment & pathSegment,
//    glm::vec3 intersect,
//    glm::vec3 normal,
//    const Material &m,
//    thrust::default_random_engine &rng)
//{
//    thrust::uniform_real_distribution<float> u01(0, 1);
//
//    glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
//    glm::vec3 newOrigin = intersect + EPSILON * normal;
//    glm::vec3 newDirection = normalize(calculateRandomDirectionInHemisphere(normal, rng));
//
//    float pdf = 1.0f;
//
//    if (m.hasRefractive > 0.0f) {
//        float eta = m.indexOfRefraction;
//        bool outside = glm::dot(incident, normal) < 0;
//        glm::vec3 n = outside ? normal : -normal;
//        float etaRatio = outside ? (1.0f / eta) : eta;
//
//        float cosTheta = glm::dot(-incident, n);
//        float sin2Theta = 1.0f - cosTheta * cosTheta;
//        bool cannotRefract = etaRatio * etaRatio * sin2Theta > 1.0f;
//
//        // Schlick's approximation
//        float r0 = (1.0f - etaRatio) / (1.0f + etaRatio);
//        r0 = r0 * r0;
//        float reflectProb = r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
//
//        if (cannotRefract || u01(rng) < reflectProb) {
//            newDirection = glm::reflect(incident, n);
//        }
//        else {
//            newDirection = glm::refract(incident, n, etaRatio);
//        }
//        attenuation = glm::vec3(1.0f);
//    }
//    else if (m.hasReflective > 0.0f) {
//        // Perfect normal reflection
//        newDirection = glm::reflect(incident, normal);
//        attenuation = m.color;
//    }
//    else {
//        // Diffuse reflection
//        float cosTheta = glm::abs(glm::dot(normal, newDirection));
//        pdf = cosTheta / PI;
//
//        attenuation = m.color;
//    }
//
//    pathSegment.ray.origin = newOrigin;
//    pathSegment.ray.direction = newDirection;
//    pathSegment.remainingBounces--;
//}
