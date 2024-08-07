set(target pbrt)
file(GLOB src
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# SET(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(${target}  ${src})
# add_subdirectory(device_code)
set_property(TARGET ${target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${target}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        >)

set(include_prefix ${CMAKE_SOURCE_DIR}/include)

target_include_directories(${target} 
            PUBLIC
            ${include_prefix}/RT
            ${include_prefix}/thirdParty
            ${include_prefix}/thirdParty/assimp-master/include
            ${include_prefix}/thirdParty/assimp-master/build/include
            ) 

set(lib_prefix ${CMAKE_SOURCE_DIR}/lib)
file(GLOB_RECURSE extern_libs ${lib_prefix}/thirdParty/*.lib)
target_link_directories(${target}
            PUBLIC
            ${lib_prefix}
            ${lib_prefix}/thirdParty
            )       

set(libs cxx glad ${extern_libs})
target_link_libraries(${target}
            PUBLIC
            ${libs}
            )

install(TARGETS ${target} DESTINATION ${lib_prefix})