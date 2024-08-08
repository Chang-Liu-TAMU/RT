set(target custom_target)

file(GLOB_RECURSE rt_include1 ${CMAKE_CURRENT_SOURCE_DIR}/include/RT/*.h)
file(GLOB_RECURSE rt_include2 ${CMAKE_CURRENT_SOURCE_DIR}/include/RT/*.cuh)
file(GLOB_RECURSE rt_include3 ${CMAKE_CURRENT_SOURCE_DIR}/include/RT/*.hpp)
add_custom_target(${target} SOURCES ${rt_include1} ${rt_include2} ${rt_include2})

