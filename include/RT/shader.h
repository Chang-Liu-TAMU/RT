#pragma once
#include "common.h"
#include "string"


class Program;

class Shader {
public:
	Shader(const char* _srcLiteral, const char* _srcFile, Program* _p,
		GLuint _t);

	GLuint getId() const {
		return _id;
	}

	std::string getType() const {
		switch (_type) {
		case GL_VERTEX_SHADER:
			return "VERTEX";
		case GL_FRAGMENT_SHADER:
			return "FRAGMENT";
		default:
			return "UNKNOWN";
		}
	}

	void check() const;
	
	void setSrcFromFiles(const char* _src);

private:
	void compile() {
		_id = glCreateShader(_type);
		glShaderSource(_id, 1, &_pSrc, NULL);
		glCompileShader(_id);
		check();
	}

	GLuint _id = GLuint(-1);
	GLuint _type = GL_VERTEX_SHADER;
	const char* _pSrc;
	Program* _pProgram;
	std::string _strShaderCode;


};