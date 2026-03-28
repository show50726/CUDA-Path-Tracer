#include "intersections.h"

// TODO: consider the scale of the box
__host__ __device__ float boxIntersectionTest(
    Instance instance,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(instance.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(instance.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(instance.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(instance.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Instance instance,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = 0.5f;

    glm::vec3 ro = multiplyMV(instance.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(instance.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(instance.transform, glm::vec4(objspaceIntersection, 1.f));
    
    normal = glm::normalize(multiplyMV(instance.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    
    return glm::length(r.origin - intersectionPoint);
}

// Reference: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__host__ __device__ float triangleIntersectionTest(
    Triangle triangle,
    Instance instance,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& outNormal)
{
    glm::vec3 ro = multiplyMV(instance.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(instance.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 edge1 = triangle.v1 - triangle.v0;
    glm::vec3 edge2 = triangle.v2 - triangle.v0;

    glm::vec3 pvec = glm::cross(rd, edge2);
    float det = glm::dot(edge1, pvec);

    if (fabs(det) < EPSILON) return -1;

    float inv_det = 1.0f / det;

    glm::vec3 tvec = ro - triangle.v0;
    float u = glm::dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return -1;

    glm::vec3 qvec = glm::cross(tvec, edge1);
    float v = glm::dot(rd, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return -1;

    float t = glm::dot(edge2, qvec) * inv_det;
    if (t < EPSILON) return -1;

    glm::vec3 objspaceIntersection = ro + rd * t;
    intersectionPoint = multiplyMV(instance.transform, glm::vec4(objspaceIntersection, 1.f));

    glm::vec3 n = glm::normalize(glm::cross(edge1, edge2));
    outNormal = glm::normalize(multiplyMV(instance.invTranspose, glm::vec4(n, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}