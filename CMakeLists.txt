cmake_minimum_required(VERSION 3.10)

project(Rt VERSION 0.1 LANGUAGES CXX C CUDA)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_subdirectory(src/pbrt)

add_subdirectory(src/cxx)
add_subdirectory(src/glad)

add_subdirectory(src/headers)


set(rt_include ${CMAKE_CURRENT_SOURCE_DIR}/include/RT)
set(target main)
add_executable(${target} 
        main.cpp)

set(include_prefix ${CMAKE_SOURCE_DIR}/include)

target_include_directories(${target} 
    PUBLIC
    ${rt_include}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/thirdParty
    ${include_prefix}/thirdParty/assimp-master/include
    ${include_prefix}/thirdParty/assimp-master/build/include
    )


file(GLOB_RECURSE extern_libs ${CMAKE_SOURCE_DIR}/lib/thirdParty/*.lib)
file(GLOB_RECURSE libs ${CMAKE_SOURCE_DIR}/lib/*.lib)
file(GLOB_RECURSE dlls ${CMAKE_SOURCE_DIR}/dlls/*.dll)

target_link_directories(${target} 
        PUBLIC
        ${CMAKE_SOURCE_DIR}/lib/thirdParty
        ${CMAKE_SOURCE_DIR}/lib
        )

target_link_libraries(${target}
        ${extern_libs}
        ${libs}
        glad
        pbrt
        cxx)
file(GLOB_RECURSE dbg_files ${CMAKE_SOURCE_DIR}/build/Debug/*.pdb)
set(des ${CMAKE_SOURCE_DIR}/bin)
install(TARGETS ${target} DESTINATION ${des})
install(TARGETS ${target} DESTINATION ${CMAKE_SOURCE_DIR})
install(FILES ${dlls} DESTINATION ${CMAKE_SOURCE_DIR})
install(FILES ${dbg_files} DESTINATION ${CMAKE_SOURCE_DIR})

    
