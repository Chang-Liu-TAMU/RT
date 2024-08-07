#pragma once
#include "common.h"
#include "mesh.h"

#define YAW -90.0f
#define PITCH 0.0f
//#define MOVE_SPEED 2.0f
#define MOVE_SPEED 100.0f
#define MOVE_SENSITIVITY 0.1f
#define ZOOM 45.0f
#define ORIGIN glm::vec3(0.0f, 0.0f, 0.0f)
#define WORLD_UP glm::vec3(0.0f, 1.0f, 0.0f)
#define FRONT glm::vec3(0.0f, 0.0f, -1.0f)
#define CROSS_AND_NORM(X, Y) glm::normalize(glm::cross(X, Y))
#define POS_X glm::vec3(3.0, 0.0, 0.0)
#define NEG_X glm::vec3(-3.0, 0.0, 0.0)
#define X_FRONT glm::vec3(-1.0, 0.0, 0.0)
#define X_YAW_POS 0.0f
#define X_YAW_NEG -180.0f
#define POS_Y glm::vec3(0.0, 3.0, 0.0)
#define NEG_Y glm::vec3(0.0, -3.0, 0.0)
#define Y_FRONT glm::vec3(0.1, -1.0, 0.1)
#define Y_YAW_POS -90.0f
#define Y_YAW_NEG -90.0f
#define POS_Z glm::vec3(0.0, 0.0, 3.0) //todo cross product with self, no Y direction now
#define NEG_Z glm::vec3(0.0, 0.0, -3.0)
#define Z_FRONT glm::vec3(0.0, 0.0, -1.0)
#define Z_YAW_POS -90.0f 
#define Z_YAW_NEG 90.0f

struct CameraState {
	glm::vec3 m_v3Position = POS_Z;
	glm::vec3 m_v3Front = FRONT;
	glm::vec3 m_v3Up;
	glm::vec3 m_v3Right;
	glm::vec3 m_v3WorldUp = WORLD_UP;

	float m_fYaw = YAW;
	float m_fPitch = PITCH;

	float m_fMoveSpeed = MOVE_SPEED;
	float m_fMouseSensitivity = MOVE_SENSITIVITY;
	float m_fZoom = ZOOM;

};

class Camera {
public:
	static Camera& getInstance() {
		static Camera* pCamera = new Camera;
		return *pCamera;
	}

	Camera(glm::vec3 _pos = POS_Z, glm::vec3 _up = WORLD_UP,
		float _yaw = YAW, float _pitch = PITCH) {
		m_v3Position = _pos;
		m_v3WorldUp = _up;
		m_fYaw = _yaw;
		m_fPitch = _pitch;
		__updateCameraVectors();
	}

	Camera(float _posX, float _posY, float _posZ, float _upX, float _upY, float _upZ,
		float _yaw, float _pitch) {
		m_v3Position = glm::vec3(_posX, _posY, _posZ);
		m_v3WorldUp = glm::vec3(_upX, _upY, _upZ);
		m_fYaw = _yaw;
		m_fPitch = _pitch;
		__updateCameraVectors();
	}

	glm::mat4 getViewMatrix() {
		_bMoved = false;
		return glm::lookAt(m_v3Position, m_v3Position + m_v3Front, m_v3Up);
	}

	void retoreFrom(CameraState _s) {
		m_v3Position = _s.m_v3Position;
		m_v3Front = _s.m_v3Front;
		m_v3Up = _s.m_v3Up;
		m_v3Right = _s.m_v3Right;
		m_v3WorldUp = _s.m_v3WorldUp;

		m_fYaw = _s.m_fYaw;
		m_fPitch = _s.m_fPitch;
		
		m_fMoveSpeed = _s.m_fMoveSpeed;
		m_fMouseSensitivity = _s.m_fMouseSensitivity;
		m_fZoom = _s.m_fZoom;

		scrolled();
		moved();
	}

	void quickShot(CameraState *_s) {
		_s->m_v3Position = m_v3Position;
		_s->m_v3Front = m_v3Front;
		_s->m_v3Up = m_v3Up;
		_s->m_v3Right = m_v3Right;
		_s->m_v3WorldUp = m_v3WorldUp;

		_s->m_fYaw = m_fYaw;
		_s->m_fPitch = m_fPitch;

		_s->m_fMoveSpeed = m_fMoveSpeed;
		_s->m_fMouseSensitivity = m_fMouseSensitivity;
	    _s->m_fZoom = m_fZoom;		
	}

	void lookFromPosX() {
		_bMoved = true;
		m_v3Position = POS_X;
		m_v3Front = X_FRONT;
		m_fYaw = X_YAW_POS;
		m_fPitch = 0.0;
	}

	void lookFromNegX() {
		_bMoved = true;
		m_v3Position = NEG_X;
		m_v3Front = -X_FRONT;
		m_fYaw = X_YAW_NEG;
		m_fPitch = 0.0;
	}

	void lookFromPosY() {
		_bMoved = true;
		m_v3Position = POS_Y;
		m_v3Front = Y_FRONT;
		m_fYaw = Y_YAW_POS;
		m_fPitch = 89.0f;
	}


	void lookFromNegY() {
		_bMoved = true;
		m_v3Position = NEG_Y;
		m_v3Front = -Y_FRONT;
		m_fYaw = Y_YAW_NEG;
		m_fPitch = -89.0;

	}

	void lookFromPosZ() {
		_bMoved = true;
		m_v3Position = POS_Z;
		m_v3Front = Z_FRONT;
		m_fYaw = Z_YAW_POS;
		m_fPitch = 0.0;
	}

	void lookFromNegZ() {
		_bMoved = true;
		m_v3Position = NEG_Z;
		m_v3Front = -Z_FRONT;
		m_fYaw = Z_YAW_NEG;
		m_fPitch = 0.0;

	}

	void moveForward(float _deltaTime) {
		m_v3Position += m_v3Front * _deltaTime * m_fMoveSpeed;
		_bMoved = true;
	}

	void moveBackward(float _deltaTime) {
		m_v3Position -= m_v3Front * _deltaTime * m_fMoveSpeed;
		_bMoved = true;
	}

	void moveLeft(float _deltaTime) {
		m_v3Position -= m_v3Right * _deltaTime * m_fMoveSpeed;
		_bMoved = true;
	}

	void moveRight(float _deltaTime) {
		m_v3Position += m_v3Right * _deltaTime * m_fMoveSpeed;
		_bMoved = true;
	}

	void moveUp(float _deltaTime) {
		m_v3Position += m_v3Up * _deltaTime * m_fMoveSpeed;
		_bMoved = true;
	}

	void moveDown(float _deltaTime) {
		m_v3Position -= m_v3Up * _deltaTime * m_fMoveSpeed;
		_bMoved = true;
	}

	void onMouseScroll(float _yOffset) {
		m_fZoom -= _yOffset;
		m_fZoom = CLIP(m_fZoom, 1.0f, 45.0f);
		_bScrolled = true;
	}

	void moved() {
		_bMoved = true;
	}
	 
	void scrolled() {
		_bScrolled = true;
	}

	bool IsScrolled() {
		if (_bScrolled) {
			_bScrolled = false;
			return true;
		}
		return false;
	}

	bool IsMoved() {
		if (_bMoved) {
			_bMoved = false;
			return true;
		}
		return false;
	}

	void lockMouseMove() {
		_bMoveLock = true;
	}

	void unlockMouseMove() {
		_bMoveLock = false;
	}

	void onMouseMove(float _xOffset, float _yOffset, GLboolean _clip = true) {
		if (_bMoveLock) {
			return;
		}

		_xOffset *= m_fMouseSensitivity;
		_yOffset *= m_fMouseSensitivity;

		m_fYaw += _xOffset;
		m_fPitch += _yOffset;

		m_fPitch = CLIP(m_fPitch, -89.0f, 89.0f);
		__updateCameraVectors();
		_bMoved = true;
	}

	void setAperture(float _aper) {
		m_fAperture = _aper;
	}

	float getAperture() {
		return m_fAperture;
	}

	void setDistToFocus(float _dtf) {
		m_fDist_to_focus = _dtf;
	}

	float getDistToFocus() {
		return m_fDist_to_focus;
	}
	
	float getDistToFocus(Mesh* _m) const {
		auto& camLoc = m_v3Position;
		auto pos = _m->getPosition();
		return glm::distance(m_v3Position, pos);
	}
	
	void setCornellCam() {
		m_v3Position = glm::vec3(278, 278, 800);
		m_fDist_to_focus = 10;
		m_fAperture = 0.0;
		m_fZoom = 40.0;
	}

	void setMoveSpeed(float _x) {
		m_fMoveSpeed = _x;
	}

	void setMouseSens(float _x) {
		m_fMouseSensitivity = _x;
	}

	void takeSnapShot() {
		std::ofstream ofs("./cache/camera", std::ios::binary);
		auto ptr = reinterpret_cast<const char*>(this);
		ofs.write(ptr, sizeof(Camera));
		ofs.flush();
		ofs.close();
	}

	void loadSnapShot() {
		std::ifstream ifs("./cache/camera", std::ios::binary);
		auto ptr = reinterpret_cast<char*>(this);
		ifs.read(ptr, sizeof(Camera));
		ifs.close();
	}

private:
	void __updateCameraVectors() {
		glm::vec3 front;
		auto cosPitch = cos(glm::radians(m_fPitch));
		auto sinPitch = sin(glm::radians(m_fPitch));
		front.x = cos(glm::radians(m_fYaw)) * cosPitch;
		front.y = sinPitch;
		front.z = sin(glm::radians(m_fYaw)) * cosPitch;
		m_v3Front = glm::normalize(front);
		m_v3Right = CROSS_AND_NORM(m_v3Front, m_v3WorldUp);
		m_v3Up = CROSS_AND_NORM(m_v3Right, m_v3Front);
	}

	bool _bMoveLock = false;
	bool _bScrolled = false;
	bool _bMoved = false;

public:
	glm::vec3 m_v3Position = POS_Z;
	glm::vec3 m_v3Front = FRONT;
	glm::vec3 m_v3Up;
	glm::vec3 m_v3Right;
	glm::vec3 m_v3WorldUp = WORLD_UP;

	float m_fYaw = YAW;
	float m_fPitch = PITCH;

	float m_fMoveSpeed = MOVE_SPEED;
	float m_fMouseSensitivity = MOVE_SENSITIVITY;
	float m_fZoom = ZOOM;

	float m_fDist_to_focus = 10.0;
	float m_fAperture = 0.0;



};
