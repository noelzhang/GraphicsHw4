set(OPENGLLIBS ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${GLEW_LIBRARIES} ${OPENGL_LIBRARY})

if(WIN32)
message(STATUS "Win32: adding glfw32dll and glew32")
set(OPENGLLIBS glfw3dll glew32 ${OPENGLLIBS})
else()
set(OPENGLLIBS glfw ${OPENGLLIBS})
endif()





set(04_srcs  04_pathtrace.cpp)                              # 04_pathtrace
add_executable(04_pathtrace ${04_srcs})                     # 04_pathtrace
target_link_libraries(04_pathtrace common ${OPENGLLIBS})    # 04_pathtrace
SOURCE_GROUP("" FILES ${04_srcs})                           # 04_pathtrace




if(CMAKE_GENERATOR STREQUAL "Xcode")
    set_property(TARGET     04_pathtrace  PROPERTY XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD c++11)
    set_property(TARGET     04_pathtrace  PROPERTY XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY libc++)
endif(CMAKE_GENERATOR STREQUAL "Xcode")


