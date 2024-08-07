#include "window.h"

bool bFirstMouse = false;
float fLastX = SCR_WIDTH / 2;
float fLastY = SCR_HEIGHT / 2;
bool bLeftButtonPressed = false;

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window, float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    auto& camera = Camera::getInstance();
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        //std::cout << "KEY(W) presseed\n";
        camera.moveForward(deltaTime);
    }
        
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        //std::cout << "KEY(S) presseed\n";
        camera.moveBackward(deltaTime);
    }
        
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        //std::cout << "KEY(A) presseed\n";
        camera.moveLeft(deltaTime);
    }
        
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        //std::cout << "KEY(D) presseed\n";
        camera.moveRight(deltaTime);
    }  

    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        //std::cout << "KEY(Q) presseed\n";
        camera.moveUp(deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
        //std::cout << "KEY(Z) presseed: lock mouse move\n";
        camera.moveDown(deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
        //std::cout << "KEY(D) presseed: lock mouse move\n";
        std::cout << "KEY(L) presseed: lock mouse move\n";
        camera.lockMouseMove();
    }

    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
        std::cout << "KEY(K) presseed: unlock mouse move\n";
        camera.unlockMouseMove();
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
//void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
//{
//    float xpos = static_cast<float>(xposIn);
//    float ypos = static_cast<float>(yposIn);
//
//    //std::cout << "xPosIn: " << xposIn << " yPosIn: " << yposIn << std::endl;
//    
//    if (bFirstMouse)
//    {
//        fLastX = xpos;
//        fLastY = ypos;
//        bFirstMouse = false;
//    }
//
//    float xoffset = xpos - fLastX;
//    float yoffset = fLastY - ypos; // reversed since y-coordinates go from bottom to top
//
//    fLastX = xpos;
//    fLastY = ypos;
//
//    Camera::getInstance().onMouseMove(xoffset, yoffset);
//}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    //std::cout << "xPosIn: " << xposIn << " yPosIn: " << yposIn << std::endl;
    if (!bLeftButtonPressed) {
        fLastX = xpos;
        fLastY = ypos;
        return;
    }

    float xoffset = xpos - fLastX;
    float yoffset = fLastY - ypos; // reversed since y-coordinates go from bottom to top

    fLastX = xpos;
    fLastY = ypos;

    Camera::getInstance().onMouseMove(xoffset, yoffset);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            bLeftButtonPressed = true;
        }
        else if (action == GLFW_RELEASE) {
            bLeftButtonPressed = false;
            
        }
    
    }
    
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    Camera::getInstance().onMouseScroll(static_cast<float>(yoffset));
}