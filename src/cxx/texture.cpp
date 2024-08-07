#include "texture.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

unsigned int TextureFromFile(const char* path, const std::string& directory, bool gamma)
{
    std::string filename = std::string(path);
    filename = directory + '/' + filename;

    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

cudaTextureObject_t cuTextureFromFile(const char* name, const std::string& directory, bool gamma) {
    std::string filename = std::string(name);
    filename = directory + '/' + filename;

    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    int texelSize = sizeof(unsigned char) * 4;

    if (nrComponents != 4) {
        throw std::exception("Error: texture channel size is not 4.\n");
    }

    //generate a texture description

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    //generate a array
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    const size_t spitch = width * texelSize;
    cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, spitch, height, cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}

unsigned char* mergeSixFaces(std::vector<unsigned char*> &_ptrs, int _size) {
    auto* ret = new unsigned char[_size * 6];
    for (int i = 0; i < 6; i++) {
        cudaMemcpy(ret + i * _size, _ptrs[i], _size * sizeof(unsigned char), cudaMemcpyHostToHost);
    }
    return ret;
}
cudaTextureObject_t cuTextureCubeMapFromFile(const char* _dir, bool gamma) {
    //std::vector<unsigned char*> sixFaces;
    //int width, height, nrComponents;
    //int texelSize = sizeof(unsigned char) * 3;
    //for (int i = 0; i < 6; i++) {
    //    std::string filePath = std::string(_dir) + "/face" + std::to_string(i) + ".png";
    //    stbi_set_flip_vertically_on_load(true);
    //    unsigned char* data = stbi_load(filePath.c_str(), &width, &height, &nrComponents, 0);
    //    sixFaces.push_back(data);
    //    if (nrComponents != 3) {
    //        throw std::exception("Error: texture channel size is not 4.\n");
    //    }
    //    if (width != height) {
    //        throw std::exception("Error: width is not equal to height.\n");
    //    }
    //}

    //auto merged = mergeSixFaces(sixFaces, width * height * 3);
    //for (auto* p : sixFaces) {
    //    delete p;
    //}
    //
    ////generate a texture description

    //cudaChannelFormatDesc channelDesc =
    //    cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindUnsigned);

    ////generate a array
    //cudaArray_t cuArray;
    //const size_t spitch = width * texelSize;
    //cudaExtent cuExt = make_cudaExtent(width, height, 6);
    //checkCudaErrors(cudaMalloc3DArray(&cuArray, &channelDesc, cuExt, cudaArrayCubemap));
    //checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, merged, spitch, spitch, height * 6, cudaMemcpyHostToDevice));
    //struct cudaResourceDesc resDesc;
    //memset(&resDesc, 0, sizeof(resDesc));
    //resDesc.resType = cudaResourceTypeArray;
    //resDesc.res.array.array = cuArray;

    //struct cudaTextureDesc texDesc;
    //memset(&texDesc, 0, sizeof(texDesc));
    //texDesc.addressMode[0] = cudaAddressModeWrap;
    //texDesc.addressMode[1] = cudaAddressModeWrap;
    //texDesc.filterMode = cudaFilterModeLinear;
    //texDesc.readMode = cudaReadModeNormalizedFloat;
    //texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    //cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}

cudaTextureObject_t* cuManyTextures(int &_num) {
    std::vector<std::string> images{ "test01.png", "test02.png", "test03.png", "test04.png"};
    auto image_num = images.size();
    cudaTextureObject_t* h_cudaTexObj = new cudaTextureObject_t[image_num];
    for (int i = 0; i < image_num; i++) {
        h_cudaTexObj[i] = cuTextureFromFile(images[i].c_str(), "images", false);
    }
    cudaTextureObject_t* d_cudaTexObj;
    checkCudaErrors(cudaMalloc(&d_cudaTexObj, sizeof(cudaTextureObject_t) * image_num));
    checkCudaErrors(cudaMemcpy(d_cudaTexObj, h_cudaTexObj, sizeof(cudaTextureObject_t) * image_num, cudaMemcpyHostToDevice));
    delete h_cudaTexObj;
    _num = image_num;
    return d_cudaTexObj;
}

CudaTexture cuTextureFromFile(const char* name, const std::string& directory) {
    std::string filename = std::string(name);
    filename = directory + '/' + filename;
    
    int width, height, nrComponents;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);

    int texelSize = sizeof(float) * 4;
    int pixelNum = height * width;
    int imgBytes = pixelNum * texelSize;

    float* data_n = new float[pixelNum * 4];

    for (int i = 0; i < pixelNum * 4; i++) {
        data_n[i] = float(data[i]) / 255.0;
    }

    if (nrComponents != 4) {
        throw std::exception("Error: texture channel size is not 4.\n");
    }

    //generate a texture description

    float* d_tex01 = nullptr;
    checkCudaErrors(cudaMalloc(&d_tex01, imgBytes));
    cudaMemcpy(d_tex01, data_n, imgBytes, cudaMemcpyHostToDevice);
    delete data;
    delete data_n;
    return CudaTexture(d_tex01, width, height);
}

Texture::Texture(std::string _name, std::string _type) {
    id = TextureFromFile(_name.c_str(), path);
    type = "texture_" + _type;
    name = _name;
}
