#include <iostream>
#include <cstring>
#include <unordered_map>
#include <stdexcept>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include "json.hpp"
#include "scene.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using json = nlohmann::json;

namespace {

bool parseMaterials(const json& materialsData, std::vector<Material>& outMaterials, 
    std::unordered_map<std::string, uint32_t>& matNameIdMap) {
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = Lambertian;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular" || p["TYPE"] == "Mirror")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = Specular;
        }
        else if (p["TYPE"] == "FresnelSpecular")
        {
            newMaterial.ior = p["IOF"];
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = FresnelSpecular;
        }
        else {
            throw std::runtime_error("Unknown material type: " + p.at("TYPE").get<std::string>());
        }

        matNameIdMap[name] = outMaterials.size();
        outMaterials.emplace_back(newMaterial);
    }

    return true;
}

bool loadObj(const std::string& objPath, std::vector<Mesh>& outMeshes, std::vector<Triangle>& outTriangles) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string err;

    std::cout << "OBJ loader starts loading OBJ from " << objPath << std::endl;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, objPath.c_str());

    if (!err.empty()) std::cout << err << std::endl;
    if (!ret)  return false;
    //if (!materials.empty()) loadObjMaterials(mtlPath, &materials);

    std::cout << "Loaded " << materials.size() << " materials." << std::endl;
    std::cout << "Loaded " << shapes.size() << " shapes." << std::endl;
    std::cout << "Loaded " << attrib.vertices.size() / 3 << " vertices." << std::endl;

    Mesh mesh;
    mesh.startIndex = outTriangles.size();
    int triangleCount = 0;
    for (auto& shape : shapes) {
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            triangleCount++;
            Triangle triangle;

            // Get three consecutive indices that form a triangle
            tinyobj::index_t idx0 = shape.mesh.indices[i];
            tinyobj::index_t idx1 = shape.mesh.indices[i + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[i + 2];

            // Get vertex positions
            triangle.v0 = glm::vec3(
                attrib.vertices[3 * idx0.vertex_index + 0],
                attrib.vertices[3 * idx0.vertex_index + 1],
                attrib.vertices[3 * idx0.vertex_index + 2]);
            triangle.v1 = glm::vec3(
                attrib.vertices[3 * idx1.vertex_index + 0],
                attrib.vertices[3 * idx1.vertex_index + 1],
                attrib.vertices[3 * idx1.vertex_index + 2]);
            triangle.v2 = glm::vec3(
                attrib.vertices[3 * idx2.vertex_index + 0],
                attrib.vertices[3 * idx2.vertex_index + 1],
                attrib.vertices[3 * idx2.vertex_index + 2]);

            // Get normals if available
            //if (idx0.normal_index >= 0 && idx1.normal_index >= 0 && idx2.normal_index >= 0) {
            //    n0 = glm::vec3(
            //        attrib.normals[3 * idx0.normal_index + 0],
            //        attrib.normals[3 * idx0.normal_index + 1],
            //        attrib.normals[3 * idx0.normal_index + 2]);
            //    n1 = glm::vec3(
            //        attrib.normals[3 * idx1.normal_index + 0],
            //        attrib.normals[3 * idx1.normal_index + 1],
            //        attrib.normals[3 * idx1.normal_index + 2]);
            //    n2 = glm::vec3(
            //        attrib.normals[3 * idx2.normal_index + 0],
            //        attrib.normals[3 * idx2.normal_index + 1],
            //        attrib.normals[3 * idx2.normal_index + 2]);
            //}
            //else {
            //    // Compute normals from the geometry of the triangle
            //    glm::vec3 edge1 = v1 - v0;
            //    glm::vec3 edge2 = v2 - v0;
            //    glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

            //    n0 = normal;
            //    n1 = normal;
            //    n2 = normal;
            //}

            outTriangles.push_back(triangle);
        }
    }
    mesh.triangleCount = triangleCount;
    outMeshes.push_back(mesh);

    return true;
}

struct GeometryDefinition {
    GeomType type;
    uint32_t index = 0;
};

using GeometryMap = std::unordered_map<uint32_t, GeometryDefinition>;

bool parseGeometries(const json& geometryData, std::vector<Triangle>& outTriangles, std::vector<Mesh>& outMeshes,
    GeometryMap& outGeometries) {
    if (!geometryData.is_array())
        throw std::runtime_error("Geometries must be an array.");
    for (const auto& p : geometryData)
    {
        const uint32_t id = p.at("ID").get<uint32_t>();
        if (outGeometries.count(id))
            throw std::runtime_error("Duplicate geometry ID: " + std::to_string(id));
        const auto type = p.at("TYPE").get<std::string>();
        GeometryDefinition geometry;
        if (type == "cube") geometry.type = CUBE;
        else if (type == "sphere") geometry.type = SPHERE;
        else if (type == "triangle") {
            geometry.type = TRIANGLE;
            geometry.index = static_cast<uint32_t>(outTriangles.size());
            Triangle triangle{};
            const auto& v0 = p.at("V0");
            const auto& v1 = p.at("V1");
            const auto& v2 = p.at("V2");
            triangle.v0 = glm::vec3(v0.at(0), v0.at(1), v0.at(2));
            triangle.v1 = glm::vec3(v1.at(0), v1.at(1), v1.at(2));
            triangle.v2 = glm::vec3(v2.at(0), v2.at(1), v2.at(2));
            triangle.n0 = triangle.n1 = triangle.n2 = glm::normalize(glm::cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
            outTriangles.push_back(triangle);
        }
        else if (type == "mesh") {
            geometry.type = MESH;
            geometry.index = static_cast<uint32_t>(outMeshes.size());
            const auto path = p.at("PATH").get<std::string>();
            if (!loadObj(path, outMeshes, outTriangles))
                throw std::runtime_error("Cannot load geometry mesh: " + path);
        }
        else throw std::runtime_error("Unknown geometry type: " + type);
        outGeometries.emplace(id, geometry);
    }
    return true;
}

bool parseInstances(const json& instanceData, const std::unordered_map<std::string, uint32_t>& matNameIdMap, const GeometryMap& geometries,
    std::vector<Instance>& outInstances) {
    for (const auto& instance : instanceData) {
        const auto geometryId = instance.at("GEOMETRY").get<uint32_t>();
        const auto geometryIt = geometries.find(geometryId);
        if (geometryIt == geometries.end())
            throw std::runtime_error("Instance references unknown geometry ID: " + std::to_string(geometryId));
        const auto matId = instance.at("MATERIAL").get<std::string>();
        const auto materialIt = matNameIdMap.find(matId);
        if (materialIt == matNameIdMap.end())
            throw std::runtime_error("Instance references unknown material: " + matId);
        const auto& translation = instance["TRANS"];
        const auto& rotation = instance["ROTAT"];
        const auto& scale = instance["SCALE"];

        const auto& geometry = geometryIt->second;
        Instance inst{};
        inst.geomType = geometry.type;
        inst.triangleId = geometry.type == TRIANGLE ? geometry.index : uint32_t(-1);
        inst.meshId = geometry.type == MESH ? geometry.index : uint32_t(-1);
        inst.materialId = materialIt->second;
        inst.translation = glm::vec3(translation[0], translation[1], translation[2]);
        inst.rotation = glm::vec3(rotation[0], rotation[1], rotation[2]);
        inst.scale = glm::vec3(scale[0], scale[1], scale[2]);
        inst.transform = utilityCore::buildTransformationMatrix(
            inst.translation, inst.rotation, inst.scale);
        inst.inverseTransform = glm::inverse(inst.transform);
        inst.invTranspose = glm::inverseTranspose(inst.transform);
        
        outInstances.push_back(std::move(inst));
    }
    return true;
}

bool parseCamera(const json& cameraData, Camera& outCamera, RenderState& outRenderState) {
    outCamera.resolution.x = cameraData["RES"][0];
    outCamera.resolution.y = cameraData["RES"][1];

    float fovy = cameraData["FOVY"];
    outRenderState.iterations = cameraData["ITERATIONS"];
    outRenderState.traceDepth = cameraData["DEPTH"];
    outRenderState.imageName = cameraData["FILE"];

    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    outCamera.position = glm::vec3(pos[0], pos[1], pos[2]);
    outCamera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    outCamera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * outCamera.resolution.x) / outCamera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    outCamera.fov = glm::vec2(fovx, fovy);

    outCamera.view = glm::normalize(outCamera.lookAt - outCamera.position);
    outCamera.right = glm::normalize(glm::cross(outCamera.view, outCamera.up));
    outCamera.pixelLength = glm::vec2(2 * xscaled / (float)outCamera.resolution.x,
        2 * yscaled / (float)outCamera.resolution.y);

    return true;
}

}

Scene::Scene(std::string filename)
{
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    const auto dot = filename.find_last_of('.');
    const auto ext = dot == std::string::npos ? std::string() : filename.substr(dot);
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        throw std::runtime_error("Expected a .json scene file: " + filename);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open scene file '" + jsonName + "'. Check that the file exists and the path is relative to the working directory.");
    }
    json data = json::parse(f);

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> matNameIdMap;
    parseMaterials(materialsData, materials, matNameIdMap);

    GeometryMap geometries;
    parseGeometries(data.at("Geometries"), triangles, meshes, geometries);
    
    const auto& instanceData = data.at("Instances");
    if (!instanceData.is_array() || instanceData.empty())
    {
        throw std::runtime_error("Scene must contain a non-empty Instances array.");
    }
    parseInstances(instanceData, matNameIdMap, geometries, instances);

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    parseCamera(cameraData, camera, state);

    // set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

