# Wannier90 Converter Tests

# Copy data directories
FILE(COPY w90_convert w90_optics DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# List of all tests
set(all_tests
    w90_convert_LaVO3
    w90_convert_SrVO3_col
    w90_convert_SrVO3_soc
    w90_convert_SrVO3_t2g_col
)

foreach(test_name ${all_tests})
  add_test(NAME Py_${test_name}
           COMMAND ${TRIQS_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT
               PYTHONPATH=${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH} ${SANITIZER_RT_PRELOAD})
endforeach()
