#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include "glm/glm.hpp"

#include "utilities.h"
#include "sceneStructs.h"
#include "materials.h"
#include "tiny_obj_loader.h"

using namespace std;

class Scene
{
public:
    Scene(string filename);
    ~Scene();

    std::vector<Mesh> meshes;
    std::vector<Triangle> triangles;
    std::vector<uint32_t> triangleToMaterialId;
    std::vector<Instance> instances;
    std::vector<Material> materials;
    RenderState state;

private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    bool loadObj(const std::string& objPath);
    bool loadObjMaterials(const std::string& objPath, std::vector<tinyobj::material_t>* materials);
};
