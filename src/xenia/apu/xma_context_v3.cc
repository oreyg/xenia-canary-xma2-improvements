/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/xma_context_v3.h"

#include <algorithm>
#include <cstring>

#include "xenia/apu/xma_frame_dumper.h"
#include "xenia/apu/xma_helpers.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/base/profiling.h"

extern "C" {
#if XE_COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable : 4101 4244 5033)
#endif
#include "third_party/FFmpeg/libavcodec/avcodec.h"
#include "third_party/FFmpeg/libavutil/channel_layout.h"
#include "third_party/FFmpeg/libavutil/error.h"
#if XE_COMPILER_MSVC
#pragma warning(pop)
#endif
}  // extern "C"

namespace xe {
namespace apu {

namespace {
constexpr int kV3IdToSampleRate[4] = {24000, 32000, 44100, 48000};
}  // namespace

XmaContextV3::XmaContextV3() = default;

XmaContextV3::~XmaContextV3() {
  if (av_context_) {
    avcodec_free_context(&av_context_);
  }
  if (av_frame_) {
    av_frame_free(&av_frame_);
  }
}

int XmaContextV3::Setup(uint32_t id, Memory* memory, uint32_t guest_ptr) {
  id_ = id;
  memory_ = memory;
  guest_ptr_ = guest_ptr;
  ResetRuntimeStateLocked();

  av_packet_ = av_packet_alloc();
  assert_not_null(av_packet_);
  av_packet_->buf = av_buffer_alloc(128 * 1024);

  av_codec_ = avcodec_find_decoder(AV_CODEC_ID_XMAFRAMES);
  if (!av_codec_) {
    XELOGE("XmaContextV3 {}: Codec not found", id);
    return 1;
  }

  av_context_ = avcodec_alloc_context3(av_codec_);
  if (!av_context_) {
    XELOGE("XmaContextV3 {}: Couldn't allocate context", id);
    return 1;
  }

  av_context_->ch_layout = AVChannelLayout{};
  av_context_->sample_rate = 0;

  av_frame_ = av_frame_alloc();
  if (!av_frame_) {
    XELOGE("XmaContextV3 {}: Couldn't allocate frame", id);
    return 1;
  }

  return 0;
}

void XmaContextV3::ResetRuntimeStateLocked() {
  xma_frame_.fill(0);
  raw_frame_.fill(0);

  current_frame_remaining_subframes_ = 0;
  remaining_subframe_blocks_in_output_buffer_ = 0;
  loop_frame_output_limit_ = 0;
  loop_start_skip_pending_ = false;

  decode_attempt_count_ = 0;
  last_input_read_offset_before_ = 0;
  last_input_read_offset_after_ = 0;
  last_current_input_packet_count_ = 0;
  last_frame_size_bits_ = 0;
  last_bits_to_copy_ = 0;
  last_next_packet_index_ = 0;
  last_current_buffer_ = 0;
  last_skip_count_ = 0;
  last_packet_index_ = -1;
  last_cross_packet_copy_ = false;
  last_swapped_input_buffer_ = false;
  last_decode_succeeded_ = false;
  last_error_status_ = 0;
}

bool XmaContextV3::Work() {
  if (!is_allocated() || !is_enabled()) {
    return false;
  }

  std::lock_guard<xe_mutex> lock(lock_);
  set_is_enabled(false);

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);
  const XMA_CONTEXT_DATA initial_data = data;

  if (!data.output_buffer_valid) {
    return true;
  }

  RingBuffer output_rb = PrepareOutputRingBuffer(&data);

  if (data.IsConsumeOnlyContext()) {
    if (current_frame_remaining_subframes_ == 0) {
      return true;
    }
    Consume(&output_rb, &data);
    data.output_buffer_write_offset =
        output_rb.write_offset() / kOutputBytesPerBlock;
    StoreContextMerged(data, initial_data, context_ptr);
    return true;
  }

  const uint32_t effective_sdc =
      std::max(static_cast<uint32_t>(1), data.subframe_decode_count);
  const int32_t minimum_subframe_decode_count =
      static_cast<int32_t>(effective_sdc) + data.output_buffer_padding;

  if (minimum_subframe_decode_count >
      remaining_subframe_blocks_in_output_buffer_) {
    StoreContextMerged(data, initial_data, context_ptr);
    return true;
  }

  while (remaining_subframe_blocks_in_output_buffer_ >=
         minimum_subframe_decode_count) {
    Decode(&data);
    Consume(&output_rb, &data);

    if (!data.IsAnyInputBufferValid() || data.error_status == 4) {
      if (data.error_status == 4) {
        XELOGW(
            "XmaContextV3 {}: decode aborted error_status=4 packet={} "
            "next={} read_before={} read_after={} frame_bits={} "
            "bits_to_copy={} skip={} cross_packet={} swapped={}",
            id(), last_packet_index_, last_next_packet_index_,
            last_input_read_offset_before_, last_input_read_offset_after_,
            last_frame_size_bits_, last_bits_to_copy_, last_skip_count_,
            last_cross_packet_copy_, last_swapped_input_buffer_);
      }
      break;
    }
  }

  data.output_buffer_write_offset =
      output_rb.write_offset() / kOutputBytesPerBlock;

  if (output_rb.empty()) {
    data.output_buffer_valid = 0;
  }

  StoreContextMerged(data, initial_data, context_ptr);
  return true;
}

void XmaContextV3::Enable() { set_is_enabled(true); }

void XmaContextV3::Clear() {
  std::lock_guard<xe_mutex> lock(lock_);

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);
  XELOGAPU(
      "XmaContextV3 {} guest-state before reset: cur_buf={} v0={} v1={} "
      "out_valid={} in_read={} out_read={} out_write={} err={}",
      id(), static_cast<uint32_t>(data.current_buffer),
      static_cast<uint32_t>(data.input_buffer_0_valid),
      static_cast<uint32_t>(data.input_buffer_1_valid),
      static_cast<uint32_t>(data.output_buffer_valid),
      static_cast<uint32_t>(data.input_buffer_read_offset),
      static_cast<uint32_t>(data.output_buffer_read_offset),
      static_cast<uint32_t>(data.output_buffer_write_offset),
      static_cast<uint32_t>(data.error_status));
  XELOGAPU(
      "XmaContextV3 {} last decode snapshot: cur_buf={} read_before={} "
      "read_after={} pkt_count={} attempt={} last_pkt={} next_pkt={} "
      "frame_bits={} bits_to_copy={} skip={} cross_packet={} swapped={} "
      "decode_ok={} last_err={}",
      id(), last_current_buffer_, last_input_read_offset_before_,
      last_input_read_offset_after_, last_current_input_packet_count_,
      decode_attempt_count_, last_packet_index_, last_next_packet_index_,
      last_frame_size_bits_, last_bits_to_copy_, last_skip_count_,
      last_cross_packet_copy_, last_swapped_input_buffer_,
      last_decode_succeeded_, last_error_status_);
  ClearLocked(&data);
  data.Store(context_ptr);
}

void XmaContextV3::ClearLocked(XMA_CONTEXT_DATA* data) {
  data->input_buffer_0_valid = 0;
  data->input_buffer_1_valid = 0;
  data->output_buffer_valid = 0;

  data->input_buffer_read_offset = kBitsPerPacketHeader;
  data->output_buffer_read_offset = 0;
  data->output_buffer_write_offset = 0;

  ResetRuntimeStateLocked();
}

void XmaContextV3::Disable() { set_is_enabled(false); }

void XmaContextV3::Release() {
  std::lock_guard<xe_mutex> lock(lock_);
  assert_true(is_allocated());

  set_is_enabled(false);
  set_is_allocated(false);
  ResetRuntimeStateLocked();
  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  std::memset(context_ptr, 0, sizeof(XMA_CONTEXT_DATA));
}

void XmaContextV3::SwapInputBuffer(XMA_CONTEXT_DATA* data) {
  if (data->current_buffer == 0) {
    data->input_buffer_0_valid = 0;
  } else {
    data->input_buffer_1_valid = 0;
  }
  data->current_buffer ^= 1;
  data->input_buffer_read_offset = kBitsPerPacketHeader;
}

int XmaContextV3::GetSampleRate(int id) {
  return kV3IdToSampleRate[std::min(id, 3)];
}

int16_t XmaContextV3::GetPacketNumber(size_t size, size_t bit_offset) {
  if (bit_offset < kBitsPerPacketHeader) {
    assert_always();
    return -1;
  }
  if (bit_offset >= (size << 3)) {
    assert_always();
    return -1;
  }
  size_t byte_offset = bit_offset >> 3;
  size_t packet_number = byte_offset / kBytesPerPacket;
  return static_cast<int16_t>(packet_number);
}

uint32_t XmaContextV3::GetCurrentInputBufferSize(XMA_CONTEXT_DATA* data) {
  return data->GetCurrentInputBufferPacketCount() * kBytesPerPacket;
}

uint8_t* XmaContextV3::GetCurrentInputBuffer(XMA_CONTEXT_DATA* data) {
  return memory()->TranslatePhysical(data->GetCurrentInputBufferAddress());
}

const uint8_t* XmaContextV3::GetNextPacket(
    XMA_CONTEXT_DATA* data, uint32_t next_packet_index,
    uint32_t current_input_packet_count) {
  if (next_packet_index < current_input_packet_count) {
    return memory()->TranslatePhysical(data->GetCurrentInputBufferAddress()) +
           next_packet_index * kBytesPerPacket;
  }

  const uint8_t next_buffer_index = data->current_buffer ^ 1;
  if (!data->IsInputBufferValid(next_buffer_index)) {
    return nullptr;
  }

  const uint32_t next_buffer_packet_count =
      data->GetInputBufferPacketCount(next_buffer_index);
  const uint32_t next_buffer_packet_index =
      next_packet_index - current_input_packet_count;
  if (next_buffer_packet_index >= next_buffer_packet_count) {
    return nullptr;
  }

  const uint32_t next_buffer_address =
      data->GetInputBufferAddress(next_buffer_index);
  if (!next_buffer_address) {
    XELOGE("XmaContextV3 {}: Buffer marked valid but null pointer!", id());
    return nullptr;
  }

  return memory()->TranslatePhysical(next_buffer_address) +
         next_buffer_packet_index * kBytesPerPacket;
}

uint32_t XmaContextV3::GetNextPacketReadOffset(
    XMA_CONTEXT_DATA* data, uint32_t next_packet_index,
    uint32_t current_input_packet_count) {
  if (next_packet_index < current_input_packet_count) {
    uint8_t* buffer =
        memory()->TranslatePhysical(data->GetCurrentInputBufferAddress());
    while (next_packet_index < current_input_packet_count) {
      uint8_t* next_packet = buffer + (next_packet_index * kBytesPerPacket);
      const uint32_t packet_frame_offset =
          xma::GetPacketFrameOffset(next_packet);
      if (packet_frame_offset <= kMaxFrameSizeinBits) {
        return (next_packet_index * kBitsPerPacket) + packet_frame_offset;
      }
      next_packet_index++;
    }
    return kBitsPerPacketHeader;
  }

  const uint8_t next_buffer_index = data->current_buffer ^ 1;
  if (!data->IsInputBufferValid(next_buffer_index)) {
    return kBitsPerPacketHeader;
  }

  const uint32_t next_buffer_address =
      data->GetInputBufferAddress(next_buffer_index);
  if (!next_buffer_address) {
    XELOGE("XmaContextV3 {}: Buffer marked valid but null pointer!", id());
    return kBitsPerPacketHeader;
  }

  uint32_t next_buffer_packet_index =
      next_packet_index - current_input_packet_count;
  const uint32_t next_buffer_packet_count =
      data->GetInputBufferPacketCount(next_buffer_index);
  uint8_t* next_buffer = memory()->TranslatePhysical(next_buffer_address);
  while (next_buffer_packet_index < next_buffer_packet_count) {
    uint8_t* next_packet =
        next_buffer + (next_buffer_packet_index * kBytesPerPacket);
    const uint32_t packet_frame_offset = xma::GetPacketFrameOffset(next_packet);
    if (packet_frame_offset <= kMaxFrameSizeinBits) {
      return (next_buffer_packet_index * kBitsPerPacket) + packet_frame_offset;
    }
    next_buffer_packet_index++;
  }

  return kBitsPerPacketHeader;
}

RingBuffer XmaContextV3::PrepareOutputRingBuffer(XMA_CONTEXT_DATA* data) {
  const uint32_t output_capacity =
      data->output_buffer_block_count * kOutputBytesPerBlock;
  const uint32_t output_read_offset =
      data->output_buffer_read_offset * kOutputBytesPerBlock;
  const uint32_t output_write_offset =
      data->output_buffer_write_offset * kOutputBytesPerBlock;

  if (output_capacity > kOutputMaxSizeBytes) {
    XELOGW(
        "XmaContextV3 {}: Output buffer exceeds expected size! "
        "(Actual: {} Max: {})",
        id(), output_capacity, kOutputMaxSizeBytes);
  }

  uint8_t* output_buffer = memory()->TranslatePhysical(data->output_buffer_ptr);

  RingBuffer output_rb(output_buffer, output_capacity);
  output_rb.set_read_offset(output_read_offset);
  output_rb.set_write_offset(output_write_offset);
  remaining_subframe_blocks_in_output_buffer_ =
      static_cast<int32_t>(output_rb.write_count()) / kOutputBytesPerBlock;

  return output_rb;
}

void XmaContextV3::StoreContextMerged(const XMA_CONTEXT_DATA& data,
                                      const XMA_CONTEXT_DATA& initial_data,
                                      uint8_t* context_ptr) {
  XMA_CONTEXT_DATA fresh(context_ptr);

  fresh.loop_count = data.loop_count;
  fresh.output_buffer_write_offset = data.output_buffer_write_offset;
  if (initial_data.input_buffer_0_valid && !data.input_buffer_0_valid) {
    fresh.input_buffer_0_valid = 0;
  }
  if (initial_data.input_buffer_1_valid && !data.input_buffer_1_valid) {
    fresh.input_buffer_1_valid = 0;
  }

  if (initial_data.output_buffer_valid && !data.output_buffer_valid) {
    fresh.output_buffer_valid = 0;
  }

  fresh.input_buffer_read_offset = data.input_buffer_read_offset;
  fresh.error_status = data.error_status;
  fresh.current_buffer = data.current_buffer;
  fresh.output_buffer_read_offset = data.output_buffer_read_offset;

  fresh.Store(context_ptr);
}

void XmaContextV3::Consume(RingBuffer* output_rb,
                           const XMA_CONTEXT_DATA* data) {
  if (!current_frame_remaining_subframes_) {
    return;
  }

  if (loop_frame_output_limit_ > 0) {
    const uint8_t total_subframes =
        (kBytesPerFrameChannel / kOutputBytesPerBlock) << data->is_stereo;
    const uint8_t consumed =
        total_subframes - current_frame_remaining_subframes_;
    if (consumed >= loop_frame_output_limit_) {
      remaining_subframe_blocks_in_output_buffer_ -=
          data->output_buffer_padding;
      current_frame_remaining_subframes_ = 0;
      loop_frame_output_limit_ = 0;
      return;
    }
  }

  const uint8_t effective_sdc =
      std::max(static_cast<uint32_t>(1), data->subframe_decode_count);
  int8_t subframes_to_write =
      std::min(static_cast<int8_t>(current_frame_remaining_subframes_),
               static_cast<int8_t>(effective_sdc));

  if (loop_frame_output_limit_ > 0) {
    const uint8_t total_subframes =
        (kBytesPerFrameChannel / kOutputBytesPerBlock) << data->is_stereo;
    const uint8_t consumed =
        total_subframes - current_frame_remaining_subframes_;
    const int8_t remaining_until_limit =
        static_cast<int8_t>(loop_frame_output_limit_ - consumed);
    if (subframes_to_write > remaining_until_limit) {
      subframes_to_write = remaining_until_limit;
    }
  }

  const int8_t raw_frame_read_offset =
      ((kBytesPerFrameChannel / kOutputBytesPerBlock) << data->is_stereo) -
      current_frame_remaining_subframes_;

  output_rb->Write(
      raw_frame_.data() + (kOutputBytesPerBlock * raw_frame_read_offset),
      subframes_to_write * kOutputBytesPerBlock);

  const int8_t headroom =
      (current_frame_remaining_subframes_ - subframes_to_write == 0)
          ? data->output_buffer_padding
          : 0;

  remaining_subframe_blocks_in_output_buffer_ -= subframes_to_write + headroom;
  current_frame_remaining_subframes_ -= subframes_to_write;
}

void XmaContextV3::UpdateLoopStatus(XMA_CONTEXT_DATA* data) {
  if (data->loop_count == 0) {
    return;
  }

  const uint32_t loop_start = std::max(kBitsPerPacketHeader, data->loop_start);
  const uint32_t loop_end = std::max(kBitsPerPacketHeader, data->loop_end);

  if (data->input_buffer_read_offset != loop_end) {
    return;
  }

  data->input_buffer_read_offset = loop_start;
  loop_start_skip_pending_ = true;

  if (data->loop_count < 255) {
    data->loop_count--;
  }
}

int XmaContextV3::PrepareDecoder(int sample_rate, bool is_two_channel) {
  sample_rate = GetSampleRate(sample_rate);
  uint32_t channels = is_two_channel ? 2 : 1;
  if (av_context_->sample_rate != sample_rate ||
      av_context_->ch_layout.nb_channels != static_cast<int>(channels)) {
    avcodec_free_context(&av_context_);
    av_context_ = avcodec_alloc_context3(av_codec_);

    av_context_->sample_rate = sample_rate;
    av_channel_layout_default(&av_context_->ch_layout, channels);
    av_context_->flags2 |= AV_CODEC_FLAG2_SKIP_MANUAL;

    if (avcodec_open2(av_context_, av_codec_, nullptr) < 0) {
      XELOGE("XmaContextV3 {}: Failed to reopen FFmpeg context", id());
      return -1;
    }
    return 1;
  }
  return 0;
}

bool XmaContextV3::DecodePacket(AVCodecContext* av_context,
                                const AVPacket* av_packet, AVFrame* av_frame) {
  auto ret = avcodec_send_packet(av_context, av_packet);
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, errbuf, sizeof(errbuf));
    XELOGE("XmaContextV3 {}: send_packet error: {} ({})", id(), errbuf, ret);
    return false;
  }
  ret = avcodec_receive_frame(av_context, av_frame);
  if (ret == AVERROR(EAGAIN)) {
    return false;
  }
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, errbuf, sizeof(errbuf));
    XELOGE("XmaContextV3 {}: receive_frame error: {} ({})", id(), errbuf, ret);
    return false;
  }
  return true;
}

void XmaContextV3::Decode(XMA_CONTEXT_DATA* data) {
  SCOPE_profile_cpu_f("apu");

  ++decode_attempt_count_;
  last_input_read_offset_before_ =
      static_cast<uint32_t>(data->input_buffer_read_offset);
  last_input_read_offset_after_ = last_input_read_offset_before_;
  last_current_input_packet_count_ = 0;
  last_frame_size_bits_ = 0;
  last_bits_to_copy_ = 0;
  last_next_packet_index_ = 0;
  last_current_buffer_ = data->current_buffer;
  last_skip_count_ = 0;
  last_packet_index_ = -1;
  last_cross_packet_copy_ = false;
  last_swapped_input_buffer_ = false;
  last_decode_succeeded_ = false;
  last_error_status_ = static_cast<uint32_t>(data->error_status);

  auto log_decode_state = [&](const char* reason) {
    XELOGW(
        "XmaContextV3 {}: {} cur_buf={} v0={} v1={} out_valid={} "
        "read_before={} read_after={} pkt_idx={} next={} pkt_count={} "
        "skip={} frame_bits={} bits_to_copy={} loop={} err={} "
        "cross_packet={} swapped={} attempt={}",
        id(), reason, static_cast<uint32_t>(data->current_buffer),
        static_cast<uint32_t>(data->input_buffer_0_valid),
        static_cast<uint32_t>(data->input_buffer_1_valid),
        static_cast<uint32_t>(data->output_buffer_valid),
        last_input_read_offset_before_, last_input_read_offset_after_,
        last_packet_index_, last_next_packet_index_,
        last_current_input_packet_count_, last_skip_count_,
        last_frame_size_bits_, last_bits_to_copy_,
        static_cast<uint32_t>(data->loop_count),
        static_cast<uint32_t>(data->error_status), last_cross_packet_copy_,
        last_swapped_input_buffer_, decode_attempt_count_);
  };

  if (!data->IsAnyInputBufferValid()) {
    return;
  }

  if (current_frame_remaining_subframes_ > 0) {
    return;
  }

  if (!data->IsCurrentInputBufferValid()) {
    last_swapped_input_buffer_ = true;
    SwapInputBuffer(data);
    if (!data->IsCurrentInputBufferValid()) {
      last_input_read_offset_after_ =
          static_cast<uint32_t>(data->input_buffer_read_offset);
      return;
    }
  }

  uint8_t* current_input_buffer = GetCurrentInputBuffer(data);

  bool is_loop_end_frame = false;
  if (data->loop_count > 0) {
    const uint32_t loop_end = std::max(kBitsPerPacketHeader, data->loop_end);
    is_loop_end_frame = (data->input_buffer_read_offset == loop_end);
  }

  UpdateLoopStatus(data);

  if (!data->output_buffer_block_count) {
    XELOGE("XmaContextV3 {}: output_buffer_block_count == 0!", id());
    return;
  }

  if (data->input_buffer_read_offset < kBitsPerPacketHeader) {
    data->input_buffer_read_offset = kBitsPerPacketHeader;
  }

  const uint32_t current_input_size = GetCurrentInputBufferSize(data);
  const uint32_t current_input_packet_count =
      current_input_size / kBytesPerPacket;
  last_current_input_packet_count_ = current_input_packet_count;

  const int16_t packet_index =
      GetPacketNumber(current_input_size, data->input_buffer_read_offset);
  last_packet_index_ = packet_index;

  if (packet_index == -1) {
    XELOGE("XmaContextV3 {}: Invalid packet index. Input read offset: {}", id(),
           static_cast<uint32_t>(data->input_buffer_read_offset));
    log_decode_state("invalid-packet-index");
    return;
  }

  auto skip_corrupt_packet = [&](const char* reason) {
    data->error_status = 4;
    last_error_status_ = static_cast<uint32_t>(data->error_status);

    const uint32_t next_packet_index = packet_index + 1;
    const bool next_packet_in_next_buffer =
        next_packet_index >= current_input_packet_count;
    uint32_t next_input_offset = GetNextPacketReadOffset(
        data, next_packet_index, current_input_packet_count);
    if (next_packet_in_next_buffer ||
        next_input_offset == kBitsPerPacketHeader) {
      last_swapped_input_buffer_ = true;
      SwapInputBuffer(data);
    }
    data->input_buffer_read_offset = next_input_offset;
    last_input_read_offset_after_ =
        static_cast<uint32_t>(data->input_buffer_read_offset);
    log_decode_state(reason);
  };

  uint8_t* packet = current_input_buffer + (packet_index * kBytesPerPacket);
  const uint32_t packet_first_frame_offset = xma::GetPacketFrameOffset(packet);
  if (packet_first_frame_offset > kMaxFrameSizeinBits) {
    skip_corrupt_packet("packet-frame-offset-invalid");
    return;
  }
  uint32_t relative_offset = data->input_buffer_read_offset % kBitsPerPacket;

  if (relative_offset < packet_first_frame_offset) {
    data->input_buffer_read_offset =
        (packet_index * kBitsPerPacket) + packet_first_frame_offset;
    relative_offset = packet_first_frame_offset;
  }

  const uint8_t skip_count = xma::GetPacketSkipCount(packet);
  last_skip_count_ = skip_count;

  if (skip_count == 0xFF) {
    const uint32_t next_packet_index = packet_index + 1;
    const bool next_packet_in_next_buffer =
        next_packet_index >= current_input_packet_count;
    uint32_t next_input_offset = GetNextPacketReadOffset(
        data, next_packet_index, current_input_packet_count);
    if (next_packet_in_next_buffer ||
        next_input_offset == kBitsPerPacketHeader) {
      last_swapped_input_buffer_ = true;
      SwapInputBuffer(data);
    }
    data->input_buffer_read_offset = next_input_offset;
    last_input_read_offset_after_ = next_input_offset;
    return;
  }

  const uint32_t next_packet_index = packet_index + skip_count + 1;
  last_next_packet_index_ = next_packet_index;

  xma::PacketWalk packet_info = xma::InspectPacket(packet, relative_offset);
  const uint8_t* next_packet =
      GetNextPacket(data, next_packet_index, current_input_packet_count);

  // Split-frame size resolution: 15-bit length prefix straddles
  // current/next packet boundary.
  if (packet_info.current_frame_size == 0) {
    if (!next_packet) {
      last_swapped_input_buffer_ = true;
      SwapInputBuffer(data);
      last_input_read_offset_after_ =
          static_cast<uint32_t>(data->input_buffer_read_offset);
      log_decode_state("missing-next-packet-for-split-frame");
      return;
    }
    last_cross_packet_copy_ = true;
    packet_info.current_frame_size =
        xma::ResolveSplitFrameSize(packet, next_packet, relative_offset);
    if (packet_info.current_frame_size == 0) {
      data->error_status = 4;
      last_error_status_ = static_cast<uint32_t>(data->error_status);
      log_decode_state("split-frame-size-invalid");
      return;
    }
  }
  last_frame_size_bits_ = packet_info.current_frame_size;

  // Frame must fit within the combined current+next payload window.
  const uint32_t combined_payload_bits =
      (kBitsPerPacket - kBitsPerPacketHeader) * 2;
  const uint32_t combined_relative_offset =
      relative_offset - kBitsPerPacketHeader;
  if (combined_relative_offset > combined_payload_bits ||
      packet_info.current_frame_size >
          (combined_payload_bits - combined_relative_offset)) {
    skip_corrupt_packet("frame-size-out-of-range");
    return;
  }
  last_bits_to_copy_ = packet_info.current_frame_size;

  // Cross-packet body: frame extends past current packet's payload end.
  // BuildAvPacketPayload pulls the overflow from next_packet, so flag
  // the diagnostic and bail if next isn't available.
  const bool needs_next_for_body =
      packet_info.current_frame_size > kBitsPerPacket - relative_offset;
  if (needs_next_for_body) {
    if (!next_packet) {
      data->error_status = 4;
      last_error_status_ = static_cast<uint32_t>(data->error_status);
      log_decode_state("missing-next-packet-last-frame");
      return;
    }
    last_cross_packet_copy_ = true;
  }

  raw_frame_.fill(0);

  const int avpacket_size = xma::BuildAvPacketPayload(
      packet, next_packet, relative_offset, packet_info.current_frame_size,
      xma_frame_.data(), static_cast<int>(xma_frame_.size()));
  if (avpacket_size <= 0) {
    skip_corrupt_packet("avpacket-build-failed");
    return;
  }

  PrepareDecoder(data->sample_rate, bool(data->is_stereo));
  av_packet_->data = xma_frame_.data();
  av_packet_->size = avpacket_size;

  XmaFrameDumper::RecordFrame(
      id(), static_cast<uint32_t>(GetSampleRate(data->sample_rate)),
      data->is_stereo ? 2 : 1, static_cast<uint8_t>(data->current_buffer),
      static_cast<uint16_t>(packet_index), data->input_buffer_read_offset,
      av_packet_->data, av_packet_->size);

  if (DecodePacket(av_context_, av_packet_, av_frame_)) {
    ConvertFrame(reinterpret_cast<const uint8_t**>(&av_frame_->data),
                 bool(data->is_stereo), raw_frame_.data());
    current_frame_remaining_subframes_ = 4 << data->is_stereo;
    last_decode_succeeded_ = true;

    if (is_loop_end_frame) {
      loop_frame_output_limit_ = (data->loop_subframe_end + 1)
                                 << data->is_stereo;
    } else {
      loop_frame_output_limit_ = 0;
    }

    if (loop_start_skip_pending_) {
      const uint8_t skip = data->loop_subframe_skip << data->is_stereo;
      if (skip < current_frame_remaining_subframes_) {
        current_frame_remaining_subframes_ -= skip;
      }
      loop_start_skip_pending_ = false;
    }
  }

  if (!packet_info.IsLastFrameInPacket()) {
    const uint32_t next_frame_offset =
        (data->input_buffer_read_offset + packet_info.current_frame_size) %
        kBitsPerPacket;
    data->input_buffer_read_offset =
        (packet_index * kBitsPerPacket) + next_frame_offset;
    last_input_read_offset_after_ =
        static_cast<uint32_t>(data->input_buffer_read_offset);
    return;
  }

  const bool next_packet_in_next_buffer =
      next_packet_index >= current_input_packet_count;
  uint32_t next_input_offset = GetNextPacketReadOffset(
      data, next_packet_index, current_input_packet_count);

  if (next_packet_in_next_buffer) {
    last_swapped_input_buffer_ = true;
    SwapInputBuffer(data);
  } else if (next_input_offset == kBitsPerPacketHeader) {
    last_swapped_input_buffer_ = true;
    SwapInputBuffer(data);
    if (data->IsAnyInputBufferValid()) {
      next_input_offset = xma::GetPacketFrameOffset(
          memory()->TranslatePhysical(data->GetCurrentInputBufferAddress()));

      if (next_input_offset > kMaxFrameSizeinBits) {
        log_decode_state("next-packet-frame-offset-invalid");
        last_swapped_input_buffer_ = true;
        SwapInputBuffer(data);
        return;
      }
    }
  }
  data->input_buffer_read_offset = next_input_offset;
  last_input_read_offset_after_ =
      static_cast<uint32_t>(data->input_buffer_read_offset);
}

}  // namespace apu
}  // namespace xe
