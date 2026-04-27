# xma-replay: offline tester for the XMA decode path.
#
# Feeds a sequence of XMA frames (the bytes that would be passed to
# avcodec_send_packet in xma_context_new.cc) through xenia's vendored
# FFmpeg and writes the decoded PCM as 16-bit RIFF/WAVE. Used to
# validate decoder changes without booting a game.

add_executable(xma-replay
  xma_replay.cc
  xma2_packet_decoder.cc
)

if(WIN32)
  target_sources(xma-replay PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_win.cc)
else()
  target_sources(xma-replay PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_posix.cc)
endif()

target_include_directories(xma-replay PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/FFmpeg
  ${PROJECT_SOURCE_DIR}/third_party/ffmpeg-xenia
)
target_link_libraries(xma-replay PRIVATE
  libavcodec
  libavutil
  xenia-apu
  xenia-base
)
xe_target_defaults(xma-replay)
set_target_properties(xma-replay PROPERTIES FOLDER "tools")
