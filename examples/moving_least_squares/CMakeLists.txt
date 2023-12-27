add_executable(ArborX_Example_MovingLeastSquares.exe moving_least_squares.cpp)
target_link_libraries(ArborX_Example_MovingLeastSquares.exe ArborX::ArborX)
add_test(NAME ArborX_Example_MovingLeastSquares COMMAND ArborX_Example_MovingLeastSquares.exe)

if(ARBORX_ENABLE_MPI)
  add_executable(ArborX_Example_DistributedMovingLeastSquares.exe distributed_moving_least_squares.cpp)
  target_link_libraries(ArborX_Example_DistributedMovingLeastSquares.exe ArborX::ArborX)
  add_test(NAME ArborX_Example_DistributedMovingLeastSquares COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ArborX_Example_DistributedMovingLeastSquares.exe> ${MPIEXEC_POSTFLAGS})
endif()
