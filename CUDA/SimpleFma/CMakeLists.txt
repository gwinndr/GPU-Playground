# GLOB C++ and CUDA sources separately
file(GLOB CUDA_APP_CPP_SOURCES 
   "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
file(GLOB CUDA_APP_CU_SOURCES 
   "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# Combine all sources
set(CUDA_APP_SOURCES ${CUDA_APP_CPP_SOURCES} ${CUDA_APP_CU_SOURCES})

add_executable(SimpleFma
   ${CUDA_APP_SOURCES})

# Link CUDA runtime
target_link_libraries(SimpleFma
    PUBLIC
        CUDA::cudart
)

if(CUDA_APP_CU_SOURCES)
    set_target_properties(SimpleFma PROPERTIES LINKER_LANGUAGE CUDA)
else()
    set_target_properties(SimpleFma PROPERTIES LINKER_LANGUAGE CXX)
endif()
