#include <iostream>
#include <cstring>
#include <unordered_map>

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
        else if (p["TYPE"] == "Specular")
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
            std::cout << "Unknown material type: " << p["TYPE"];
            return false;
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
    mesh.triangleCount = attrib.vertices.size() / 3;
    mesh.startIndex = outTriangles.size();
    outMeshes.push_back(mesh);
    for (auto& shape : shapes) {
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
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

    return true;
}

bool parseGeometries(const json& geometryData, std::vector<Triangle>& outTriangles, std::vector<Mesh>& outMeshes,
    std::unordered_map<uint32_t, uint32_t>& outGeomIdIndexMap) {
    for (const auto& p : geometryData)
    {
        const auto& id = p["ID"];
        const auto& material = p["MATERIAL"];
        const auto& path = p["PATH"];
        Mesh mesh;
        outGeomIdIndexMap[id] = outMeshes.size();
        loadObj(path, outMeshes, outTriangles);
    }
    return true;
}

bool parseInstances(const json& instanceData, std::unordered_map<std::string, uint32_t>& matNameIdMap, std::unordered_map<uint32_t, uint32_t>& geomIdIndexMap,
    std::vector<Instance>& outInstances, std::vector<Triangle>& outTriangles) {
    for (const auto& instance : instanceData) {
        const auto& type = instance["TYPE"];
        const auto& matId = instance["MATERIAL"];
        const auto& translation = instance["TRANS"];
        const auto& rotation = instance["ROTAT"];
        const auto& scale = instance["SCALE"];

        GeomType geomType = CUBE;
        if (type == "cube") geomType = CUBE;
        else if (type == "sphere") geomType = SPHERE;
        else if (type == "triangle") geomType = TRIANGLE;
        else if (type == "mesh") geomType = MESH;

        Instance inst;
        if (geomType == TRIANGLE) {
            const auto& v0 = instance["V0"];
            const auto& v1 = instance["V1"];
            const auto& v2 = instance["V2"];
            Triangle triangle;
            triangle.v0 = glm::vec3(v0[0], v0[1], v0[2]);
            triangle.v1 = glm::vec3(v1[0], v1[1], v1[2]);
            triangle.v2 = glm::vec3(v2[0], v2[1], v2[2]);

            inst.triangleId = outTriangles.size();
            outTriangles.push_back(triangle);
        }

        inst.geomType = geomType;
        inst.meshId = geomType == MESH ? geomIdIndexMap[instance["MESHID"]] : -1;
        inst.materialId = matNameIdMap[matId];
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

    outCamera.right = glm::normalize(glm::cross(outCamera.view, outCamera.up));
    outCamera.pixelLength = glm::vec2(2 * xscaled / (float)outCamera.resolution.x,
        2 * yscaled / (float)outCamera.resolution.y);

    outCamera.view = glm::normalize(outCamera.lookAt - outCamera.position);

    return true;
}

}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> matNameIdMap;
    parseMaterials(materialsData, materials, matNameIdMap);

    const auto& geometryData = data["Geometries"];
    std::unordered_map<uint32_t, uint32_t> geomIdIndexMap;
    parseGeometries(geometryData, triangles, meshes, geomIdIndexMap);
    
    const auto& instanceData = data["Instances"];
    parseInstances(instanceData, matNameIdMap, geomIdIndexMap, instances, triangles);

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    parseCamera(cameraData, camera, state);

    // set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

