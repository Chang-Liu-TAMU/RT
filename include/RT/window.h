#pragma once
#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <functional>
#include <utility>
#include <mutex>

#include "common.h"
#include "program.h"
#include "mesh.h"
#include "texture.h"
#include "camera.h"


// settings
#define SCR_WIDTH (1280 / 4)
#define SCR_HEIGHT (720 / 4)
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window, float deltaTime=0);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);


extern bool bFirstMouse;
extern float fLastX;
extern float fLastY;

class Window {
public:
	Window(unsigned int _w, unsigned int _h, const char* _title = "LearnOpenGL", unsigned int _fps = 30)
		: _uiFps(_fps), _llFrameTime(1000 / _fps) {
		// glfw: initialize and configure
		// ------------------------------
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

		_pWindow = glfwCreateWindow(_w, _h, _title, NULL, NULL);
		if (_pWindow == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			_bInitSuccess = false;
			return;
		}
		glfwMakeContextCurrent(_pWindow);
		glfwSetFramebufferSizeCallback(_pWindow, framebuffer_size_callback);
		glfwSetCursorPosCallback(_pWindow, mouse_callback);
		glfwSetScrollCallback(_pWindow, scroll_callback);
		glfwSetMouseButtonCallback(_pWindow, mouse_button_callback);

		// glad: load all OpenGL function pointers
		// ---------------------------------------
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
			_bInitSuccess = false;
			return;
		}
	}

	~Window() {
		glfwTerminate();
	}

	void setMeshesLock(std::mutex *_p) {
		m_mtxMeshesLock = _p;
	}

	void renderLoop() {
		// render loop
		// -----------
		float radianPerSec = 10.0;
		float k = 0.3;

		float deltaTime = 0.0f;
		float lastFrame = 0.0f;
		
		Camera& cam = Camera::getInstance();
		//cam.setCornellCam();
		glm::mat4 projection = glm::perspective(glm::radians(cam.m_fZoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 3000.0f);
		glm::mat4 view = cam.getViewMatrix();


		glEnable(GL_DEPTH_TEST);
		glEnable(GL_STENCIL_TEST);
		glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

		auto& allMeshes = *m_mapAllMeshes;
		auto* light = allMeshes["light"].get();
		while (!glfwWindowShouldClose(_pWindow))
		{
			
			// input
			// -----
			float currTime = glfwGetTime();
			deltaTime = currTime - lastFrame;
			lastFrame = currTime;

			processInput(_pWindow, deltaTime);

			auto frameStartTime = std::chrono::high_resolution_clock::now();

			// render
			// ------
			glStencilMask(0xff);
			glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			
			if (cam.IsScrolled()) {
				projection = glm::perspective(glm::radians(cam.m_fZoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 3000.0f);
			}

			// camera/view transformation
			if (cam.IsMoved())
			{
				view = cam.getViewMatrix();
			}

			checkGlError("draw\n");
			{
				std::lock_guard<std::mutex> lg(*m_mtxMeshesLock);
				for (auto& [name, v] : allMeshes) {
					v->Draw(projection, view, light);
				}
			}
			
			/*for (auto* p : m_vecMeshRawPtrs) {
				p->Draw(projection, view, light);
			}*/
			//lightingShader->use();
			//lightingShader->setProj(projection);
			//lightingShader->setView(view);
			//lightingShader->setVec3("objectColor", 1.0f, 0.5f, 0.31f);
			//lightingShader->setVec3("lightColor", 1.0f, 1.0f, 1.0f);
			//glm::mat4 model = glm::mat4(1.0f);
			//lightingShader->setModel(model);
			//
			//glDrawArrays(GL_TRIANGLES, 0, 36);

			//lightCubeShader->use();
			//lightingShader->setVec3("lightColor", 1.0f, 1.0f, 1.0f);
			//lightCubeShader->setProj(projection);
			//lightCubeShader->setView(view);
			//model = glm::mat4(1.0f);
			//model = glm::translate(model, lightPos);
			//model = glm::scale(model, glm::vec3(0.2f)); // a smaller cube
			//lightCubeShader->setModel(model);
			//glDrawArrays(GL_TRIANGLES, 0, 36);

			//glDrawArrays(GL_TRIANGLES, 0, 3);
			//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
			// -------------------------------------------------------------------------------
			
			glfwSwapBuffers(_pWindow);
			glfwPollEvents();
			
			auto frameEndTime = std::chrono::high_resolution_clock::now();
			auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(frameEndTime - frameStartTime).count();
			if (timeDiff < _llFrameTime) {
				std::this_thread::sleep_for(std::chrono::milliseconds(_llFrameTime - timeDiff));
			}
		}
	}

	bool initSuccess() {
		return _bInitSuccess;
	}


	void traverseMesh(std::function<void(Mesh*)> _f) {
		for (auto &[_, v] : *m_mapAllMeshes) {
			_f(v.get());
		}
	}


	GLFWwindow *_pWindow = nullptr;
	bool _bInitSuccess = true;

	unsigned int _uiFps = 60;
	long long _llFrameTime = 0;

	std::mutex* m_mtxMeshesLock = nullptr;

	std::map<std::string, MESH_PTR>* m_mapAllMeshes = nullptr;

	void setMeshesSrc(decltype(m_mapAllMeshes) _p) {
		m_mapAllMeshes = _p;
	}
	
};