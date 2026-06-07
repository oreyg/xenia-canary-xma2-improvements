add_library(xenia-loc STATIC)
xe_platform_sources(xenia-loc ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xenia-loc PUBLIC
  fmt
  xenia-base
  xenia-cpu
)
# Only the stackpoint-walk diagnostics ([sp]/buffer-discover) touch the x64
# backend; the rest of the module is backend-agnostic.
if(XE_TARGET_X86_64)
  target_link_libraries(xenia-loc PUBLIC xenia-cpu-backend-x64)
endif()
xe_target_defaults(xenia-loc)
