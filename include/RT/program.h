#pragma once
#include "common.h"
#include "shader.h"
#include "texture.h"
#include <vector>

class Program {
public:
	Program() {
		_id = glCreateProgram();
	}

	void check(const Shader* _s) const {
		int success;
		char infoLog[512];
		glGetShaderiv(_s->getId(), GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(_s->getId(), 512, NULL, infoLog);
			std::cout << "ERROR::SHADER" + _s->getType() + ":COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		checkGlError("6");
	}

	void check() {
		int success;
		char infoLog[512];
		glGetProgramiv(_id, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(_id, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		}
	}

	GLuint getId() {
		return _id;
	}

	void link() {
		
		for (auto* s : _m_vecShaderPtrs) {
			glAttachShader(_id, s->getId());
		}
		
		glLinkProgram(_id);
		checkGlError("4");
		check();
		checkGlError("5");
	}

	void use() {
		glUseProgram(_id);
	}

	void setShader(Shader* _s) {
		_m_vecShaderPtrs.push_back(_s);
	}
	
	GLint getLocation(const std::string& name) const {
		return glGetUniformLocation(_id, name.c_str());
	}

public:
	//set uniforms
	void setBool(const std::string& name, bool value) const {
		glUniform1i(getLocation(name), (int)value);
	}

	void setInt(const std::string& name, int value) const {
		glUniform1i(getLocation(name), value);
	}

	void setFloat(const std::string& name, float value) const {
		glUniform1f(getLocation(name), value);
	}

	void setVec2(const std::string& name, const glm::vec2 &value) const {
		glUniform2fv(getLocation(name), 1, &value[0]);
	}
	
	void setVec2(const std::string& name, float x, float y) const {
		glUniform2f(getLocation(name), x, y);
	}

	void setVec3(const std::string& name, const glm::vec3& value) const {
		glUniform3fv(getLocation(name), 1, &value[0]);
	}

	void setVec3(const std::string& name, float x, float y, float z) const
	{
		glUniform3f(getLocation(name), x, y, z);
	}

	void setVec4(const std::string& name, const glm::vec4& value) const
	{
		glUniform4fv(getLocation(name), 1, &value[0]);
	}

	void setVec4(const std::string& name, float x, float y, float z, float w) const
	{
		glUniform4f(getLocation(name), x, y, z, w);
	}

	void setMat2(const std::string& name, const glm::mat2& mat) const
	{
		glUniformMatrix2fv(getLocation(name), 1, GL_FALSE, &mat[0][0]);
	}
	
	void setMat3(const std::string& name, const glm::mat3& mat) const
	{
		glUniformMatrix3fv(getLocation(name), 1, GL_FALSE, &mat[0][0]);
	}

	void setMat4(const std::string& name, const glm::mat4& mat) const
	{
		glUniformMatrix4fv(getLocation(name), 1, GL_FALSE, &mat[0][0]);
	}

	void setProj(const glm::mat4& mat) const {
		setMat4("projection", mat);
	}

	void setView(const glm::mat4& mat) const {
		setMat4("view", mat);
	}

	void setModel(const glm::mat4& mat) const {
		setMat4("model", mat);
	}

private:
	GLuint _id;
	std::vector<Shader*> _m_vecShaderPtrs;
};