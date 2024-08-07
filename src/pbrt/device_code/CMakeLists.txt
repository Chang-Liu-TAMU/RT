set(target device_code)
file(GLOB_RECURSE src
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${target} ${src})

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

set(libs ${extern_libs})
target_link_libraries(${target}
            PUBLIC
            ${libs}
            )

install(TARGETS ${target} DESTINATION ${lib_prefix})