#include "program.h"

Shader::Shader(const char* _srcLiteral, const char* _srcFile, Program* _p,
	GLuint _t) : _pProgram(_p), _type(_t) {
	if (_srcLiteral) {
		_pSrc = _srcLiteral;
	}
	else {
		setSrcFromFiles(_srcFile);
	}

	if (!_pSrc) {
		throw std::exception("Error: shader has no source !!!");
	}
	compile();
	_p->setShader(this);
}

void Shader::check() const {
	_pProgram->check(this);
}

void Shader::setSrcFromFiles(const char* _src) {
	std::ifstream shaderFile;
	//ensure ifstream objects can throw exceptions
	shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try {
		shaderFile.open(_src);
		std::stringstream shaderStream;
		shaderStream << shaderFile.rdbuf();
		shaderFile.close();
		_strShaderCode = shaderStream.str();
	}
	catch (std::ifstream::failure& e) {
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
	}
	_pSrc = _strShaderCode.c_str();
}




