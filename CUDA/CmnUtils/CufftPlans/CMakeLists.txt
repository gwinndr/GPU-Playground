# GLOB C++ and CUDA sources separately
file(GLOB CUDA_APP_CPP_SOURCES 
   "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
file(GLOB CUDA_APP_CU_SOURCES 
   "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)
file(GLOB APP_HEADERS 
   "${CMAKE_CURRENT_SOURCE_DIR}/inc/*"
)

# Combine all sources
set(CUDA_APP_SOURCES ${CUDA_APP_CPP_SOURCES} ${CUDA_APP_CU_SOURCES})

add_library(CufftPlans SHARED
   ${CUDA_APP_SOURCES} ${APP_HEADERS})

# Link CUDA runtime
target_link_libraries(CufftPlans
    PUBLIC
        CUDA::cudart
        CUDA::cufft
)

target_include_directories(
   CufftPlans
   PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

if(CUDA_APP_CU_SOURCES)
    set_target_properties(CufftPlans PROPERTIES LINKER_LANGUAGE CUDA)
else()
    set_target_properties(CufftPlans PROPERTIES LINKER_LANGUAGE CXX)
endif()
