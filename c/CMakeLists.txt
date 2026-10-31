cmake_minimum_required(VERSION 3.16)
project(jamdetect C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# The library is the code that would run on the controller: float32 only,
# no heap, no libm. Warnings are errors.
add_library(jamdetect
    src/jd_math.c
    src/jd_mras.c
    src/jd_emf.c
    src/jd_detector.c)
target_include_directories(jamdetect PUBLIC include)
target_compile_options(jamdetect PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion -Wshadow
    -Werror -fno-math-errno)

option(JD_BUILD_TESTS "Build host tests" ON)
if(JD_BUILD_TESTS)
    enable_testing()
    add_executable(test_vectors test/test_vectors.c)
    target_link_libraries(test_vectors PRIVATE jamdetect m)
    target_compile_options(test_vectors PRIVATE -Wall -Wextra -Werror)
    target_compile_definitions(test_vectors PRIVATE
        JD_VECTOR_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vectors")
    add_test(NAME vectors COMMAND test_vectors)
endif()
