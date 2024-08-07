//hello triangle
#include <thread>
#include <map>
#include <exception>
#include <sstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <time.h>

#include "window.h"
#include "model.h"
#include "MeshFactory.h"
#include "pbrt_engine.h"

#define DIELECTRIC 7
#define LAMBERTIAN 10
#define METAL 6
#define LIGHT 3

Program* createProgram(const char *_vs, const char *_fs) {
    Program* pProgram = new Program();
    Shader* vShader = new Shader(nullptr, _vs, pProgram, GL_VERTEX_SHADER);
    Shader* fShader = new Shader(nullptr, _fs, pProgram, GL_FRAGMENT_SHADER);

    pProgram->link();

    return pProgram;
}

//load model, use camera view, simple light
//
static std::map<std::string, MESH_PTR> meshesCollection;
static std::mutex meshCollectionMtx;

static int currWinWidth = SCR_WIDTH;
static int currWinHeight = SCR_HEIGHT;
static CameraState currCamState;
static Program* commonShader = nullptr;
void addXyPlane(float w = 1.0, float h = 1.0, int _texIdx = 0, bool _flip = false) {
    auto ptr = make_a_xy_plane(w, h, _flip);
    ptr->setProgram(commonShader);
    //ptr->setMatType(1);
    ptr->setMatType(LAMBERTIAN);
    ptr->registerThisModel(meshesCollection);
    ptr->setTexIdx(_texIdx);
}

void addYzPlane(float w = 1.0, float h = 1.0, int _texIdx = 0, bool _flip = false) {
    auto ptr = make_a_yz_plane(w, h, _flip);
    ptr->setProgram(commonShader);
    ptr->setMatType(LAMBERTIAN);
    ptr->registerThisModel(meshesCollection); 
    ptr->setTexIdx(_texIdx);
}

void addZxPlane(float w = 1.0, float h = 1.0, int _texIdx = 0, bool _flip = false) {
    auto ptr = make_a_zx_plane(w, h, _flip);
    ptr->setProgram(commonShader);
    ptr->setMatType(LAMBERTIAN);
    ptr->registerThisModel(meshesCollection);
    ptr->setTexIdx(_texIdx);
}

void console(Window *_pWin) {
    Mesh* curr = nullptr;
    char input[20];
    std::string cache;
    std::vector<std::string> tokens;
    bool loop = true;
    auto &engine = PbrtEngine::getInstance();
    auto& cam = Camera::getInstance();
    while (loop) {
        tokens.clear();
        try {
            std::cout << ">>>:";
            std::cin.getline(input, 30); //len matters, dead loop figure out why?
            cache = std::string(input);
            std::istringstream iss(cache);
            std::string token;
            while (iss >> token) {
                tokens.push_back(token);
            }
            if (tokens.empty()) {
                continue;
            }
            /*int x = 4;
            while (x--) {
                tokens.push_back("-999");
            }*/
            auto first = tokens[0];

            if (first == "q") {
                std::cout << "bye, press any buttion to end.\n";
                int i = 0;
                loop = false;
                std::cin >> i;
            }

            if (first == "m") {
                auto model = tokens[1];
                if (meshesCollection.find(model) == meshesCollection.end()) {
                    std::cout << "ERROR: Model " << model << " not found.\n";
                    continue;
                }
                auto *new_mesh = meshesCollection[model].get();
                /*if (curr) {
                    curr->outlineOff();
                }*/
                curr = new_mesh;
                //curr->outlineOn();
                std::cout << "setting model finished.\n";
                continue;
            }

            if (first == "camera") {
                auto second = tokens[1];
                if (second == "-h") {
                    std::cout << "[options]: posx, negx, log, restore, info, aperture, \n"
                        "dist_to_focus, speed, sens" << std::endl;
                }
                if (second == "posx") {
                    Camera::getInstance().lookFromPosX();
                }
                else if (second == "negx") {
                    Camera::getInstance().lookFromNegX();
                }
                else if (second == "posy") {
                    Camera::getInstance().lookFromPosY();
                }
                else if (second == "negy") {
                    Camera::getInstance().lookFromNegY();
                }
                else if (second == "posz") {
                    Camera::getInstance().lookFromPosZ();
                }
                else if (second == "negz") {
                    Camera::getInstance().lookFromNegZ();
                }
                else if (second == "log") {
                    Camera::getInstance().quickShot(&currCamState);
                    std::cout << "camera quick shot finished.\n";
                }
                else if (second == "restore") {
                    Camera::getInstance().retoreFrom(currCamState);
                    std::cout << "camera state restored.\n";
                }
                else if (second == "info") {
                    std::cout << Camera::getInstance().m_v3Position << std::endl;
                    std::cout << Camera::getInstance().m_v3Front << std::endl;
                }
                else if (second == "aperture") {
                    Camera::getInstance().setAperture(std::stof(tokens[2]));
                }
                else if (second == "dist_to_focus") {
                    if (tokens.size() > 2) {
                        Camera::getInstance().setDistToFocus(std::stof(tokens[2]));
                    }
                    else {
                        auto f = cam.getDistToFocus(curr);
                        cam.setDistToFocus(f);
                        std::cout << "INFO: set camera distance_to_focus to " << f << std::endl;
                    }

                }
                else if (second == "speed") {
                    if (tokens.size() < 3) {
                        std::cout << "curr camera speed: " << cam.m_fMoveSpeed << std::endl;
                        continue;
                    }
                    cam.setMoveSpeed(std::stof(tokens[2]));
                }
                else if (second == "sens") {
                    cam.setMouseSens(std::stof(tokens[2]));
                }
                else if (second == "save") {
                    cam.takeSnapShot();
                }
                else if (second == "load") {
                    cam.loadSnapShot();
                }
                continue;
            }

            if (first == "render") {
                if (tokens.size() < 2) {
                    goto RENDER;
                }
                if (tokens[1] == "status") {
                    engine.checkStatus();
                }
                else if (tokens[1] == "param") {
                    engine.setNx(std::stoi(tokens[2]));
                    engine.setNy(std::stoi(tokens[3]));
                    engine.setNs(std::stoi(tokens[4]));
                    if (engine.nXnYChanged()) {
                        engine.uponPixelNumChange();
                    }
                }
                else if (tokens[1] == "hd") {
                    engine.setNx(1280);
                    engine.setNy(720);
                    engine.setNs(100);
                    if (engine.nXnYChanged()) {
                        engine.uponPixelNumChange();
                    }
                }
                else {
                RENDER:
                    bool buildScene = true;
                    if (tokens.size() >= 2 && tokens[1] == "-cache") {
                        buildScene = false;
                    }
                    std::lock_guard<std::mutex> lg(*_pWin->m_mtxMeshesLock);
                    engine.renderScene(buildScene);
                    engine.join();
                }
                continue;
            }

            if (!curr) {
                continue;
            }

            if (first == "mx") {
                curr->translateXCommand(std::stof(tokens[1]));
                continue;
            }

            if (first == "my") {
                curr->translateYCommand(std::stof(tokens[1]));
                continue;
            }

            if (first == "mz") {
                curr->translateZCommand(std::stof(tokens[1]));
                continue;
            }

            if (first == "rx") {
                curr->rotationXCommand(std::stof(tokens[1]));
                continue;
            }

            if (first == "ry") {
                curr->rotationYCommand(std::stof(tokens[1]));
                continue;
            }
            if (first == "rz") {
                curr->rotationZCommand(std::stof(tokens[1]));
                continue;
            }

            if (first == "s") {
                curr->scaleCommand(std::stof(tokens[1]));
                continue;
            }

            if (first == "log") {
                if (tokens[1] == "position") {
                    if (tokens.size() < 3) {
                        curr->positionSnapshot();
                    }
                    else if (tokens[2] == "all") {
                        for (auto& [k, v] : meshesCollection) {
                            v->positionSnapshot();
                        }
                    }
                    
                }

            }

            if (first == "ambient") {
                curr->ambientCommand(std::stof(tokens[1]),
                    std::stof(tokens[2]), std::stof(tokens[3]));
            }

            if (first == "specular") {
                curr->specularCommand(std::stof(tokens[1]),
                    std::stof(tokens[2]), std::stof(tokens[3]));
            }

            if (first == "diffuse") {
                curr->diffuseCommand(std::stof(tokens[1]),
                    std::stof(tokens[2]), std::stof(tokens[3]));
            }

            if (first == "reset") {
                curr->resetAll();
                continue;
            }

            if (first == "hide") {
                curr->undisplay();
            }

            if (first == "show") {
                curr->display();
            }

            if (first == "viewport") {
                std::cout << "curr window info: " << "width " << currWinWidth << " | " << "height " << currWinWidth << std::endl;
            }

            if (first == "new") {
                if (tokens[1] == "xy" || tokens[1] == "yx") {
                    addXyPlane();
                }
                else if (tokens[1] == "yz" || tokens[1] == "zy") {
                    addYzPlane();
                }
                else if (tokens[1] == "zx" || tokens[1] == "xz") {
                    addZxPlane();
                }
            }

            if (first == "radius") {
                std::cout << "radius => " << curr->getAverageRadius() << std::endl;
            }

        }
        catch (std::exception& e) {
            std::cout << "Console Error: " << e.what() << std::endl;
        }
    }

    std::cout << "Quitting this thread: " << std::this_thread::get_id() << std::endl;
}

MESH_PTR signature_reader() {
    std::string path = ".\\cache\\cube_locations.txt";
    std::ifstream ifs(path);
    std::string line;
    float width = 0.0;
    std::vector<std::pair<float, float>> locations;
    std::getline(ifs, line);
    width = std::stof(line);
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        std::string loc;
        std::vector<float> v;
        while (iss >> loc) {
            v.push_back(std::stof(loc));
        }
        if (v.size() != 2) {
            throw std::exception("failed in signature reader.");
        }
        locations.push_back({ v[0], v[1] });
    }
    auto meshList = std::make_shared<Mesh>();
    float scale = 300;
    float hw = width / 2.5;
    srand(time(nullptr));
    double r = float(rand()) / float(RAND_MAX);
    for (auto& zx : locations) {
        auto x = zx.second;
        auto z = zx.first;
        float h = width * (r * 2.0f + 1.0);
        auto box02 = make_a_cube_mesh(vec3(x - hw, 0, z - hw), vec3(x + hw, h, z + hw), 0.0, vec3(0.0, 0.0, 0.0), false);
        box02->shareMeshData(meshList);
    }
    meshList->setupMesh();
    meshList->setName("meshList");
    return meshList;
}

int main() {
    //test
    /*std::map<int, int> m{ {1, 1} };
    auto it = m.try_emplace(1, 2);
    std::cout << it.second << " " << it.first->first << " " << it.first->second << std::endl;
    it = m.try_emplace(2, 3);
    std::cout << it.second << " " << it.first->first << " " << it.first->second << std::endl;
    return 0;*/

    //test
	auto *pWin = new Window(SCR_WIDTH, SCR_HEIGHT);
    pWin->setMeshesLock(&meshCollectionMtx);
	if (!pWin->initSuccess()) {
		std::cout << "Error: GLFW initalization error\n";
		return -1;
	}
    
    pWin->setMeshesSrc(&meshesCollection);

    auto* pProgram01 = createProgram("shaders//lighting_maps.vs\0", "shaders//lighting_maps.fs\0");
    
    auto* pProgram02 = createProgram("shaders//light_cube.vs\0", "shaders//light_cube.fs\0");

    auto* pProgram03 = createProgram("shaders//lighting_with_texture.vs\0", "shaders//lighting_with_texture.fs\0");

    auto* outlineShader = createProgram("shaders//outline.vs\0", "shaders//outline.fs\0");
    
    commonShader = pProgram03;

    //cornell box
    /*auto box01 = make_a_cube_mesh(vec3(0, 0, 0), vec3(165, 165, 165), -18, vec3(269, 0, 295)); 
    box01->setName("box01"); 
    box01->setProgramSubMesh(pProgram03);
    box01->setMatTypeSubMesh(METAL);
    box01->registerThisModel(meshesCollection);*/
    
    //############################################# start
    auto light = make_a_cornell_light();
    light->setName("light");
    light->setMatType(LIGHT);
    light->registerThisModel(meshesCollection);
    light->setProgram(pProgram02);
    light->setOutlineShader(outlineShader);

    /*auto box02 = make_a_cube_mesh(vec3(0, 0, 0), vec3(165, 330, 165), 15, vec3(130, 0, 65), true);
    box02->setMatTypeSubMesh(METAL);
    box02->setProgramSubMesh(pProgram03);
    box02->setName("box02");
    box02->registerThisModel(meshesCollection);
    box02->setOutlineShader(outlineShader);*/
    //
    
    auto model1 = Model("models/teapot.obj", "t1", 1.0f);
    auto mesh1 = model1.getFristMesh();
    mesh1->setProgram(pProgram02);
    mesh1->setMatType(5);
    mesh1->registerThisModel(meshesCollection);

    /*auto model2 = Model("models/bunny.obj", "b1", 1.0f);
    auto mesh2 = model2.getFristMesh();
    mesh2->setProgram(pProgram01);
    mesh2->setMatType(4);
    mesh2->registerThisModel(meshesCollection);*/

    auto model3 = Model("models/cube.obj", "cube", 1.0f);
    auto mesh3 = model3.getFristMesh();
    /*mesh3->textures.clear();
    mesh3->textures.push_back(Texture::getTexture("test01.png"));*/
    mesh3->setMatType(8);
    //mesh3->setTexIdx(0);
    //mesh3->setMatType(DIELECTRIC);//���⣺ lambertian ��work ?? 
    mesh3->setProgram(pProgram01);
    mesh3->registerThisModel(meshesCollection);
    mesh3->setIsSphere();
    //mesh3->setOutlineShader(outlineShader);

    //many spheres
    auto localRadom = []() -> float {
        return 2 * (drand48() - 0.5);
    };
    std::vector<vec3> all_centers {vec3(0, -1000, 0), vec3(0, 1, 0), vec3(-4, 1, 0), vec3(4, 1, 0)};
    std::vector<float> all_radius {1000.0, 1.0, 1.0, 1.0};
    //std::vector<int> all_mats {0, 7, 1, 6};
    std::vector<int> all_mats {0, 7, 4, 6};
    std::vector<bool> vec_flags(4, true);
    int step = 2;
    for (int a = -11; a < 11; a+=step) {
        for (int b = -11; b < 11; b+=step) {
            float choose_mat = drand48();
            auto r = 0.2 + drand48() / 10;
            vec3 center(a + 0.2 * localRadom(), r, b + 0.2 * localRadom());
            if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
                all_centers.push_back(center);
                all_radius.push_back(r);
                if (drand48() < 0.5)
                    vec_flags.push_back(false);
                else
                    vec_flags.push_back(true);
                if (choose_mat < 0.3) {
                    //diffuse
                    all_mats.push_back(10 + drand48() * 20);
                }
                else if (choose_mat < 0.9) {
                    all_mats.push_back(30 + drand48() * 20);
                }
                else {
                    all_mats.push_back(7);
                }

            }
        }
    }

    for (int i = 0; i < all_centers.size(); i++) {
        auto mesh = std::make_shared<FlyWeight>(mesh3);
        auto center = all_centers[i];
        auto r = all_radius[i];
        auto mat = all_mats[i];
        auto flag = vec_flags[i];
        mesh->setName("c" + std::to_string(i));
        mesh->setMatType(mat);
        mesh->setProgram(pProgram01);
        mesh->setPositionFromVec3(center);
        mesh->m_fScale = r;
        mesh->registerThisModel(meshesCollection);
        if (flag) {
            mesh->setIsSphere();
        }
        
    }
    
    //many spheres

    //auto mesh4 = std::make_shared<FlyWeight>(mesh3);
    //mesh4->setName("c1");
    //mesh4->setMatType(4);
    //mesh4->setProgram(pProgram01);
    //mesh4->registerThisModel(meshesCollection);
    //mesh4->setIsSphere();

    //auto mesh5 = std::make_shared<FlyWeight>(mesh3);
    //mesh5->setName("c2");
    ////mesh5->setMatType(9);
    //mesh5->setMatType(2);
    ////mesh5->setMatType(5);
    //mesh5->setProgram(pProgram01);
    //mesh5->registerThisModel(meshesCollection);
    //mesh5->setIsSphere();
    //
    //auto mesh6 = std::make_shared<FlyWeight>(mesh3);
    //mesh6->setName("c3");
    //mesh6->setMatType(5);
    //mesh6->setProgram(pProgram01);
    //mesh6->registerThisModel(meshesCollection);
    //mesh6->setIsSphere();
    //

    //auto floor = make_a_zx_plane(-500, -500, 500, 500, 0);
    //floor->setName("floor");
    //floor->registerThisModel(meshesCollection);
    //floor->setProgram(pProgram01);
    //floor->setMatType(0);
    //floor->setOutlineShader(outlineShader);
    //floor->setMatType(0);

#ifdef X
    /*auto model4= Model("models/girl.obj", "girl", 1.0f);
    auto mesh4 = model4.getFristMesh();
    mesh4->textures.clear();
    mesh4->textures.push_back(Texture::getTexture("test01.png"));
    mesh4->setMatType(LAMBERTIAN);
    mesh4->setProgram(pProgram03);
    mesh4->registerThisModel(meshesCollection);
    mesh4->setTexIdx(0);
    mesh4->setOutlineShader(outlineShader);*/
    //mesh4->setIsSphere();

    
    
    auto left_wall = make_a_yz_plane(0, 0, 555, 555, 0, true);
    left_wall->setName("left_wall");
    left_wall->registerThisModel(meshesCollection);
    left_wall->setProgram(pProgram01);
    left_wall->setMatType(8);
    left_wall->setOutlineShader(outlineShader);
    //left_wall->setMatType(LAMBERTIAN);
    //left_wall->setMatType(5);
    //left_wall->setTexIdx(1);

    auto right_wall = make_a_yz_plane(0, 0, 555, 555, 555);
    right_wall->setName("right_wall");
    right_wall->registerThisModel(meshesCollection);
    right_wall->setProgram(pProgram01);
    right_wall->setMatType(9);
    right_wall->setOutlineShader(outlineShader);
    //right_wall->setMatType(LAMBERTIAN);

    auto ceil = make_a_zx_plane(0, 0, 555, 555, 555, true);
    ceil->setName("ceil");
    ceil->registerThisModel(meshesCollection);
    ceil->setProgram(pProgram01);
    ceil->setMatType(10);
    ceil->setOutlineShader(outlineShader);

    auto floor = make_a_zx_plane(0, 0, 555, 555, 0);
    floor->setName("floor");
    floor->registerThisModel(meshesCollection);
    floor->setProgram(pProgram01);
    floor->setMatType(10);
    floor->setOutlineShader(outlineShader);
    //floor->setMatType(0);

    auto back_wall = make_a_xy_plane(0, 0, 555, 555, 0);
    back_wall->setName("back_wall");
    back_wall->registerThisModel(meshesCollection);
    back_wall->setProgram(pProgram01);
    back_wall->setMatType(10);
    back_wall->setOutlineShader(outlineShader);
    //back_wall->setMatType(LAMBERTIAN);
    //back_wall->setTexIdx(1);

    //addXyPlane(10.0, 10.0, 3, true); //���⣺ this is not okay why normal ���˲�work ???
    //############################################# end
#endif
    //#########################     new start
#ifdef CAT
    auto box02 = make_a_cube_mesh(vec3(0, 0, 0), vec3(800, 450, 450), 15, vec3(0, 0, 0), true);
    box02->setMatTypeSubMesh(LIGHT);
    box02->setProgramSubMesh(pProgram03);
    box02->setName("box02");
    box02->registerThisModel(meshesCollection);
    //
    auto model3 = Model("models/sphere.obj", "s1", 1.0f);
    auto mesh3 = model3.getFristMesh();
    mesh3->setMatType(5);
    //mesh3->setMatType(DIELECTRIC);//���⣺ lambertian ��work ?? 
    mesh3->setProgram(pProgram01);
    mesh3->registerThisModel(meshesCollection);
    mesh3->setIsSphere();

    int height = 800;
    auto left_wall = make_a_yz_plane(-1000, 0, 1000, height, -1000, true, true);
    left_wall->setName("left_wall");
    left_wall->registerThisModel(meshesCollection);
    left_wall->setProgram(pProgram01);
    //left_wall->setMatType(8);
    left_wall->setMatType(4);
    //left_wall->setMatType(5);
    left_wall->setTexIdx(1);

    auto right_wall = make_a_yz_plane(-1000, 0, 1000, height, 1000);
    right_wall->setName("right_wall");
    right_wall->registerThisModel(meshesCollection);
    right_wall->setProgram(pProgram01);
    //right_wall->setMatType(9);
    right_wall->setMatType(LIGHT);
    right_wall->setTexIdx(2);

    /*auto ceil = make_a_zx_plane(-1000, -1000, 1000, 1000, height, true);
    ceil->setName("ceil");
    ceil->registerThisModel(meshesCollection);
    ceil->setProgram(pProgram01);
    ceil->setMatType(10);*/

    auto back_wall = make_a_xy_plane(-1000, 0, 1000, height, -1000);
    back_wall->setName("back_wall");
    back_wall->registerThisModel(meshesCollection);
    back_wall->setProgram(pProgram01);
    //back_wall->setMatType(10);
    back_wall->setMatType(LIGHT);
    back_wall->setTexIdx(0);

    auto light = make_a_cornell_light();
    light->setName("light");
    light->setMatType(LIGHT);
    light->registerThisModel(meshesCollection);
    light->setProgram(pProgram02);

    auto m = signature_reader();
    m->setMatType(METAL);
    m->registerThisModel(meshesCollection);
    m->setProgram(pProgram01);

    auto floor = make_a_zx_plane(-1000, -1000, 1000, 1000, 0, false, true);
    floor->setName("floor");
    floor->registerThisModel(meshesCollection);
    floor->setProgram(pProgram01);
    floor->setMatType(10);
    floor->setMatType(0);

    //######################## new end
#endif
    Camera::getInstance().loadSnapShot();
   //cornel box

    /*auto model1 = Model("models/monkey.obj", "m1", 1.0f);
    auto mesh1 = model1.getFristMesh();
    mesh1->setProgram(pProgram01);
    mesh1->setMatType(1);
    mesh1->registerThisModel(meshesCollection);*/

    /*auto mesh2SP = make_a_light(2.0, 2.0);
    auto* mesh2 = mesh2SP.get();
    mesh2->setName("light");
    mesh2->setMatType(LIGHT);
    mesh2->registerThisModel(meshesCollection);
    mesh2->setProgram(pProgram02);*/

    /*auto model3 = Model("models/cube.obj", "cube", 1.0f);
    auto mesh3 = model3.getFristMesh();
    mesh3->setMatType(METAL);
    mesh3->setProgram(pProgram01);
    mesh3->registerThisModel(meshesCollection);*/

    //auto model4 = Model("models/cube.obj", "", 1.0f);
    //auto mesh4 = model4.getFristMesh();
    ////mesh4->setIsSphere();
    //mesh4->setMatType(METAL);
    //mesh4->setProgram(pProgram01);
    //mesh4->registerThisModel(meshesCollection);

    /*addXyPlane(1.0, 1.0, 0);
    addYzPlane(1.0, 1.0, 1);
    addZxPlane(1.0, 1.0, 2);
    addXyPlane(1.0, 1.0, 0, true);
    addYzPlane(1.0, 1.0, 1, true);
    addZxPlane(1.0, 1.0, 2, true);*/

    

    //auto back_wall = make_a_xy_plane(0, 0, 555, 555, 0);
    //back_wall->setName("back_wall");
    //back_wall->registerThisModel(meshesCollection);
    //back_wall->setProgram(pProgram01);
    ////back_wall->setMatType(10);
    //back_wall->setMatType(LAMBERTIAN);
    //back_wall->setTexIdx(3);

    pWin->traverseMesh([](Mesh* _p) { _p->loadSnapshot(); });

    for (auto &[_, p] : meshesCollection) {
        PbrtEngine::getInstance().addMesh(p.get());
    }

    PbrtEngine::getInstance().genTextures();

    std::thread cons(console, pWin);

	pWin->renderLoop();

    //todo why crash if we do not have this thing
    if (cons.joinable()) {
        cons.join();
    }

    delete pWin;
	return 0;
}