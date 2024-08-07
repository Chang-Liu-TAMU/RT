#pragma once
#include "mesh.h"

inline std::shared_ptr<LightMesh> make_a_light(int w = 2.0, int h = 2.0, int y = 0.0) {
	decltype(Mesh::vertices) vertcies(4, Vertex());
	Vertex& v1 = vertcies[0];
	Vertex& v2 = vertcies[1];
	Vertex& v3 = vertcies[2];
	Vertex& v4 = vertcies[3];
	v1.Position = glm::vec3(-w, y, -h);
	v2.Position = glm::vec3(-w, y, h);
	v3.Position = glm::vec3(w, y, h);
	v4.Position = glm::vec3(w, y, -h);

	v1.Normal = glm::vec3(0.0, -1.0, 0.0);
	v2.Normal = glm::vec3(0.0, -1.0, 0.0);
	v3.Normal = glm::vec3(0.0, -1.0, 0.0);
	v4.Normal = glm::vec3(0.0, -1.0, 0.0);

	decltype(Mesh::indices) indices{ 1, 0, 3, 1, 3, 2 };

	decltype(Mesh::textures) textures;

	auto ptr = std::make_shared<LightMesh>(std::move(vertcies), std::move(indices), std::move(textures));
	ptr->setupMesh();
	return ptr;
}

inline std::shared_ptr<Mesh> make_a_light(float lbz, float lbx, float rtz, float rtx, float y = 1.0, bool _flip = false) {
	decltype(Mesh::vertices) vertcies(4, Vertex());
	Vertex& v1 = vertcies[0];
	Vertex& v2 = vertcies[1];
	Vertex& v3 = vertcies[2];
	Vertex& v4 = vertcies[3];

	v1.Position = glm::vec3(rtx, y, lbz);
	v2.Position = glm::vec3(lbx, y, lbz);
	v3.Position = glm::vec3(lbx, y, rtz);
	v4.Position = glm::vec3(rtx, y, rtz);

	v1.Normal = glm::vec3(0.0, 1.0, 0.0);
	v2.Normal = glm::vec3(0.0, 1.0, 0.0);
	v3.Normal = glm::vec3(0.0, 1.0, 0.0);
	v4.Normal = glm::vec3(0.0, 1.0, 0.0);

	v1.TexCoords = glm::vec2(0.0, 1.0);
	v2.TexCoords = glm::vec2(0.0, 0.0);
	v3.TexCoords = glm::vec2(1.0, 0.0);
	v4.TexCoords = glm::vec2(1.0, 1.0);


	if (_flip) {
		v1.Normal = glm::vec3(0.0, -1.0, 0.0);
		v2.Normal = glm::vec3(0.0, -1.0, 0.0);
		v3.Normal = glm::vec3(0.0, -1.0, 0.0);
		v4.Normal = glm::vec3(0.0, -1.0, 0.0);

		v1.TexCoords = glm::vec2(1.0, 1.0);
		v2.TexCoords = glm::vec2(1.0, 0.0);
		v3.TexCoords = glm::vec2(0.0, 0.0);
		v4.TexCoords = glm::vec2(0.0, 1.0);
	}

	decltype(Mesh::indices) indices{ 0, 1, 2, 0, 2, 3 };
	if (_flip) {
		indices = { 3, 2, 1, 3, 1, 0 };
	}

	decltype(Mesh::textures) textures{ };

	auto ptr = std::make_shared<LightMesh>(std::move(vertcies), std::move(indices), std::move(textures));
	ptr->setupMesh();
	return ptr;
}

inline std::shared_ptr<Mesh> make_a_xy_plane(float w = 1.0, float h = 1.0, bool _flip = false) {
	decltype(Mesh::vertices) vertcies(4, Vertex());
	Vertex& v1 = vertcies[0];
	Vertex& v2 = vertcies[1];
	Vertex& v3 = vertcies[2];
	Vertex& v4 = vertcies[3];
	float half_w = w / 2;
	float half_h = h / 2;
	v1.Position = glm::vec3(-half_w, half_h, 0.0);
	v2.Position = glm::vec3(-half_w, -half_h, 0.0);
	v3.Position = glm::vec3(half_w, -half_h, 0.0);
	v4.Position = glm::vec3(half_w, half_h, 0.0);

	v1.Normal = glm::vec3(0.0, 0.0, 1.0);
	v2.Normal = glm::vec3(0.0, 0.0, 1.0);
	v3.Normal = glm::vec3(0.0, 0.0, 1.0);
	v4.Normal = glm::vec3(0.0, 0.0, 1.0);
	
	v1.TexCoords = glm::vec2(0.0, 1.0);
	v2.TexCoords = glm::vec2(0.0, 0.0);
	v3.TexCoords = glm::vec2(1.0, 0.0);
	v4.TexCoords = glm::vec2(1.0, 1.0);


	if (_flip) {
		v1.Normal = glm::vec3(0.0, 0.0, -1.0);
		v2.Normal = glm::vec3(0.0, 0.0, -1.0);
		v3.Normal = glm::vec3(0.0, 0.0, -1.0);
		v4.Normal = glm::vec3(0.0, 0.0, -1.0);

		v1.TexCoords = glm::vec2(1.0, 1.0);
		v2.TexCoords = glm::vec2(1.0, 0.0);
		v3.TexCoords = glm::vec2(0.0, 0.0);
		v4.TexCoords = glm::vec2(0.0, 1.0);
	}

	decltype(Mesh::indices) indices{ 0, 1, 2, 0, 2, 3 };
	if (_flip) {
		indices = {3, 2, 1, 3, 1, 0};
	}

	decltype(Mesh::textures) textures { Texture("test01.png") };

	float dummy = _flip ? 2.0 : 1.0;
	auto ptr = std::make_shared<Mesh>(std::move(vertcies), std::move(indices), std::move(textures));
	ptr->setupMesh();
	ptr->setName(_flip ? "xy_flip": "xy");
	ptr->setDiffuse(glm::vec3(1.0 / dummy, 0.0, 0.0));

	return ptr;
}

inline std::shared_ptr<Mesh> make_a_xy_plane(float lbx, float lby, float rtx, float rty, float z, bool _flip = false, bool _setup = true) {
	decltype(Mesh::vertices) vertcies(4, Vertex());
	Vertex& v1 = vertcies[0];
	Vertex& v2 = vertcies[1];
	Vertex& v3 = vertcies[2];
	Vertex& v4 = vertcies[3];

	v1.Position = glm::vec3(lbx, rty, z);
	v2.Position = glm::vec3(lbx, lby, z);
	v3.Position = glm::vec3(rtx, lby, z);
	v4.Position = glm::vec3(rtx, rty, z);

	v1.Normal = glm::vec3(0.0, 0.0, 1.0);
	v2.Normal = glm::vec3(0.0, 0.0, 1.0);
	v3.Normal = glm::vec3(0.0, 0.0, 1.0);
	v4.Normal = glm::vec3(0.0, 0.0, 1.0);

	v1.TexCoords = glm::vec2(0.0, 1.0);
	v2.TexCoords = glm::vec2(0.0, 0.0);
	v3.TexCoords = glm::vec2(1.0, 0.0);
	v4.TexCoords = glm::vec2(1.0, 1.0);


	if (_flip) {
		v1.Normal = glm::vec3(0.0, 0.0, -1.0);
		v2.Normal = glm::vec3(0.0, 0.0, -1.0);
		v3.Normal = glm::vec3(0.0, 0.0, -1.0);
		v4.Normal = glm::vec3(0.0, 0.0, -1.0);

		v1.TexCoords = glm::vec2(1.0, 1.0);
		v2.TexCoords = glm::vec2(1.0, 0.0);
		v3.TexCoords = glm::vec2(0.0, 0.0);
		v4.TexCoords = glm::vec2(0.0, 1.0);
	}

	decltype(Mesh::indices) indices{ 0, 1, 2, 0, 2, 3 };
	if (_flip) {
		indices = { 3, 2, 1, 3, 1, 0 };
	}

	decltype(Mesh::textures) textures{ Texture::getTexture("test01.png") };

	float dummy = _flip ? 2.0 : 1.0;
	auto ptr = std::make_shared<Mesh>(std::move(vertcies), std::move(indices), std::move(textures));
	if (_setup) {
		ptr->setupMesh();
		ptr->setName(_flip ? "xy_flip" : "xy");
		ptr->setDiffuse(glm::vec3(1.0 / dummy, 0.0, 0.0));
	}
	
	return ptr;
}

inline std::shared_ptr<Mesh> make_a_yz_plane(float w = 1.0, float h = 1.0, bool _flip = false) {
	decltype(Mesh::vertices) vertcies(4, Vertex());
	Vertex& v1 = vertcies[0];
	Vertex& v2 = vertcies[1];
	Vertex& v3 = vertcies[2];
	Vertex& v4 = vertcies[3];
	float half_w = w / 2;
	float half_h = h / 2;
	v1.Position = glm::vec3(0.0, half_h, -half_w);
	v2.Position = glm::vec3(0.0, -half_h, -half_w);
	v3.Position = glm::vec3(0.0, -half_h, half_w);
	v4.Position = glm::vec3(0.0, half_h, half_w);

	v1.Normal = glm::vec3(-1.0, 0.0, 0.0);
	v2.Normal = glm::vec3(-1.0, 0.0, 0.0);
	v3.Normal = glm::vec3(-1.0, 0.0, 0.0);
	v4.Normal = glm::vec3(-1.0, 0.0, 0.0);

	v1.TexCoords = glm::vec2(0.0, 1.0);
	v2.TexCoords = glm::vec2(0.0, 0.0);
	v3.TexCoords = glm::vec2(1.0, 0.0);
	v4.TexCoords = glm::vec2(1.0, 1.0);


	if (_flip) {
		v1.Normal = glm::vec3(1.0, 0.0, 0.0);
		v2.Normal = glm::vec3(1.0, 0.0, 0.0);
		v3.Normal = glm::vec3(1.0, 0.0, 0.0);
		v4.Normal = glm::vec3(1.0, 0.0, 0.0);

		v1.TexCoords = glm::vec2(1.0, 1.0);
		v2.TexCoords = glm::vec2(1.0, 0.0);
		v3.TexCoords = glm::vec2(0.0, 0.0);
		v4.TexCoords = glm::vec2(0.0, 1.0);
	}

	decltype(Mesh::indices) indices{ 0, 1, 2, 0, 2, 3 };
	if (_flip) {
		indices = { 3, 2, 1, 3, 1, 0 };
	}

	decltype(Mesh::textures) textures { Texture("test02.png") };

	float dummy = _flip ? 2.0 : 1.0;
	auto ptr = std::make_shared<Mesh>(std::move(vertcies), std::move(indices), std::move(textures));
	ptr->setupMesh();
	ptr->setName(_flip ? "yz_flip" : "yz");
	ptr->setDiffuse(glm::vec3(0.0, 1.0 / dummy, 0.0));
	return ptr;
}

inline std::shared_ptr<Mesh> make_a_yz_plane(float lbz, float lby, float rtz, float rty, float x, bool _flip = false, bool _setup = true) {
	decltype(Mesh::vertices) vertcies(4, Vertex());
	Vertex& v1 = vertcies[0];
	Vertex& v2 = vertcies[1];
	Vertex& v3 = vertcies[2];
	Vertex& v4 = vertcies[3];

	v1.Position = glm::vec3(x, rty, lbz);
	v2.Position = glm::vec3(x, lby, lbz);
	v3.Position = glm::vec3(x, lby, rtz);
	v4.Position = glm::vec3(x, rty, rtz);

	v1.Normal = glm::vec3(-1.0, 0.0, 0.0);
	v2.Normal = glm::vec3(-1.0, 0.0, 0.0);
	v3.Normal = glm::vec3(-1.0, 0.0, 0.0);
	v4.Normal = glm::vec3(-1.0, 0.0, 0.0);

	v1.TexCoords = glm::vec2(0.0, 1.0);
	v2.TexCoords = glm::vec2(0.0, 0.0);
	v3.TexCoords = glm::vec2(1.0, 0.0);
	v4.TexCoords = glm::vec2(1.0, 1.0);


	if (_flip) {
		v1.Normal = glm::vec3(1.0, 0.0, 0.0);
		v2.Normal = glm::vec3(1.0, 0.0, 0.0);
		v3.Normal = glm::vec3(1.0, 0.0, 0.0);
		v4.Normal = glm::vec3(1.0, 0.0, 0.0);

		v1.TexCoords = glm::vec2(1.0, 1.0);
		v2.TexCoords = glm::vec2(1.0, 0.0);
		v3.TexCoords = glm::vec2(0.0, 0.0);
		v4.TexCoords = glm::vec2(0.0, 1.0);
	}

	decltype(Mesh::indices) indices{ 0, 1, 2, 0, 2, 3 };
	if (_flip) {
		indices = { 3, 2, 1, 3, 1, 0 };
	}

	decltype(Mesh::textures) textures{ Texture::getTexture("test02.png") };

	float dummy = _flip ? 2.0 : 1.0;
	auto ptr = std::make_shared<Mesh>(std::move(vertcies), std::move(indices), std::move(textures));
	if (_setup) {
		ptr->setupMesh();
		ptr->setName(_flip ? "yz_flip" : "yz");
		ptr->setDiffuse(glm::vec3(0.0, 1.0 / dummy, 0.0));
	}
	
	return ptr;
}

inline std::shared_ptr<Mesh> make_a_zx_plane(float w = 1.0, float h = 1.0, bool _flip = false) {
	decltype(Mesh::vertices) vertcies(4, Vertex());
	Vertex& v1 = vertcies[0];
	Vertex& v2 = vertcies[1];
	Vertex& v3 = vertcies[2];
	Vertex& v4 = vertcies[3];
	float half_w = w / 2;
	float half_h = h / 2;
	v1.Position = glm::vec3(half_h, 0.0, -half_w);
	v2.Position = glm::vec3(-half_h, 0.0, -half_w);
	v3.Position = glm::vec3(-half_h, 0.0, half_w);
	v4.Position = glm::vec3(half_h, 0.0, half_w);

	v1.Normal = glm::vec3(0.0, 1.0, 0.0);
	v2.Normal = glm::vec3(0.0, 1.0, 0.0);
	v3.Normal = glm::vec3(0.0, 1.0, 0.0);
	v4.Normal = glm::vec3(0.0, 1.0, 0.0);

	v1.TexCoords = glm::vec2(0.0, 1.0);
	v2.TexCoords = glm::vec2(0.0, 0.0);
	v3.TexCoords = glm::vec2(1.0, 0.0);
	v4.TexCoords = glm::vec2(1.0, 1.0);


	if (_flip) {
		v1.Normal = glm::vec3(0.0, -1.0, 0.0);
		v2.Normal = glm::vec3(0.0, -1.0, 0.0);
		v3.Normal = glm::vec3(0.0, -1.0, 0.0);
		v4.Normal = glm::vec3(0.0, -1.0, 0.0);

		v1.TexCoords = glm::vec2(1.0, 1.0);
		v2.TexCoords = glm::vec2(1.0, 0.0);
		v3.TexCoords = glm::vec2(0.0, 0.0);
		v4.TexCoords = glm::vec2(0.0, 1.0);
	}

	decltype(Mesh::indices) indices{ 0, 1, 2, 0, 2, 3 };
	if (_flip) {
		indices = { 3, 2, 1, 3, 1, 0 };
	}

	decltype(Mesh::textures) textures { Texture("test03.png") };

	float dummy = _flip ? 2.0 : 1.0;
	auto ptr = std::make_shared<Mesh>(std::move(vertcies), std::move(indices), std::move(textures));
	ptr->setupMesh();
	ptr->setName(_flip ? "zx_flip" : "zx");
	ptr->setDiffuse(glm::vec3(0.0, 0.0, 1.0 / dummy));
	return ptr;
}


inline std::shared_ptr<Mesh> make_a_zx_plane(float lbz, float lbx, float rtz, float rtx, float y = 1.0, bool _flip = false, bool _setup = true) {
	decltype(Mesh::vertices) vertcies(4, Vertex());
	Vertex& v1 = vertcies[0];
	Vertex& v2 = vertcies[1];
	Vertex& v3 = vertcies[2];
	Vertex& v4 = vertcies[3];

	v1.Position = glm::vec3(rtx, y, lbz);
	v2.Position = glm::vec3(lbx, y, lbz);
	v3.Position = glm::vec3(lbx, y, rtz);
	v4.Position = glm::vec3(rtx, y, rtz);

	v1.Normal = glm::vec3(0.0, 1.0, 0.0);
	v2.Normal = glm::vec3(0.0, 1.0, 0.0);
	v3.Normal = glm::vec3(0.0, 1.0, 0.0);
	v4.Normal = glm::vec3(0.0, 1.0, 0.0);


	v1.TexCoords = glm::vec2(0.0, 1.0);
	v2.TexCoords = glm::vec2(0.0, 0.0);
	v3.TexCoords = glm::vec2(1.0, 0.0);
	v4.TexCoords = glm::vec2(1.0, 1.0);


	if (_flip) {
		v1.Normal = glm::vec3(0.0, -1.0, 0.0);
		v2.Normal = glm::vec3(0.0, -1.0, 0.0);
		v3.Normal = glm::vec3(0.0, -1.0, 0.0);
		v4.Normal = glm::vec3(0.0, -1.0, 0.0);

		v1.TexCoords = glm::vec2(1.0, 1.0);
		v2.TexCoords = glm::vec2(1.0, 0.0);
		v3.TexCoords = glm::vec2(0.0, 0.0);
		v4.TexCoords = glm::vec2(0.0, 1.0);
	}

	decltype(Mesh::indices) indices{ 0, 1, 2, 0, 2, 3 };
	if (_flip) {
		indices = { 3, 2, 1, 3, 1, 0 };
	}

	decltype(Mesh::textures) textures{ Texture::getTexture("test03.png") };

	float dummy = _flip ? 2.0 : 1.0;
	auto ptr = std::make_shared<Mesh>(std::move(vertcies), std::move(indices), std::move(textures));
	if (_setup) {
		ptr->setupMesh();
		ptr->setName(_flip ? "zx_flip" : "zx");
		ptr->setDiffuse(glm::vec3(0.0, 0.0, 1.0 / dummy));
	}
	
	return ptr;
}

MESH_PTR make_a_cube_mesh(vec3 lb, vec3 rt, float ry, vec3 translation, bool _setup) {
	auto front = make_a_xy_plane(lb[0], lb[1], rt[0], rt[1], rt[2], false, _setup);
	auto back = make_a_xy_plane(lb[0], lb[1], rt[0], rt[1], lb[2], true, _setup);

	auto left = make_a_yz_plane(lb[2], lb[1], rt[2], rt[1], lb[0], false, _setup);
	auto right = make_a_yz_plane(lb[2], lb[1], rt[2], rt[1], rt[0], true, _setup);

	auto top = make_a_zx_plane(lb[2], lb[0], rt[2], rt[0], rt[1], false, _setup);
	auto bottom = make_a_zx_plane(lb[2], lb[0], rt[2], rt[0], lb[1], true, _setup);

	MESH_PTR cube = std::make_shared<CubeMesh>(front, back, left, right, top, bottom);
	cube->rotationYCommand(ry);
	cube->translateCommand(translation);
	
	front->m_ptrTopMesh = cube;
	back->m_ptrTopMesh = cube;
	left->m_ptrTopMesh = cube;
	right->m_ptrTopMesh = cube;
	top->m_ptrTopMesh = cube;
	bottom->m_ptrTopMesh = cube;

	front->setTexIdx(3);
	back->setTexIdx(3);
	left->setTexIdx(3);
	right->setTexIdx(3);
	top->setTexIdx(3);
	bottom->setTexIdx(3);
	
	return cube;

}

MESH_PTR make_a_cornell_light() {
	float diff = 0;
	int h = 10;
	//h = 780;
	auto m = make_a_light(-4 - diff, -4 - diff,  4 + diff, 4 + diff, h, true);
	return m;
}
