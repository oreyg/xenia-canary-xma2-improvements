/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APU_XMA_CONTEXT_V3_H_
#define XENIA_APU_XMA_CONTEXT_V3_H_

#include <array>
#include <atomic>
#include <mutex>

#include "xenia/apu/xma_context.h"
#include "xenia/apu/xma_frame_walker.h"
#include "xenia/base/bit_stream.h"
#include "xenia/base/ring_buffer.h"
#include "xenia/memory.h"
#include "xenia/xbox.h"

struct AVCodec;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;

namespace xe {
namespace apu {

// Same overall shape as XmaContextNew (AV_CODEC_ID_XMAFRAMES, bit-level walk,
// Decode/Consume split) but adopts fixes from AC6 RECOMP. Mainly:
//   - skip_corrupt_packet recovery on bad packet headers / split frames
//   - GetNextPacket / GetNextPacketReadOffset crossing buffer boundaries
class XmaContextV3 : public XmaContext {
 public:
  static constexpr uint32_t kBitsPerPacketHeader = 32;
  static constexpr uint32_t kMaxFrameSizeinBits = 0x4000 - kBitsPerPacketHeader;

  explicit XmaContextV3();
  ~XmaContextV3() override;

  int Setup(uint32_t id, Memory* memory, uint32_t guest_ptr) override;
  bool Work() override;

  void Enable() override;
  void Clear() override;
  void Disable() override;
  void Release() override;

 private:
  void ClearLocked(XMA_CONTEXT_DATA* data);
  void ResetRuntimeStateLocked();

  static void SwapInputBuffer(XMA_CONTEXT_DATA* data);
  static int GetSampleRate(int id);
  static int16_t GetPacketNumber(size_t size, size_t bit_offset);
  static uint32_t GetCurrentInputBufferSize(XMA_CONTEXT_DATA* data);

  const uint8_t* GetNextPacket(XMA_CONTEXT_DATA* data,
                               uint32_t next_packet_index,
                               uint32_t current_input_packet_count);
  uint32_t GetNextPacketReadOffset(XMA_CONTEXT_DATA* data,
                                   uint32_t next_packet_index,
                                   uint32_t current_input_packet_count);
  uint8_t* GetCurrentInputBuffer(XMA_CONTEXT_DATA* data);

  void Decode(XMA_CONTEXT_DATA* data);
  void Consume(RingBuffer* output_rb, const XMA_CONTEXT_DATA* data);
  void UpdateLoopStatus(XMA_CONTEXT_DATA* data);

  RingBuffer PrepareOutputRingBuffer(XMA_CONTEXT_DATA* data);

  int PrepareDecoder(int sample_rate, bool is_two_channel);
  bool DecodePacket(AVCodecContext* av_context, const AVPacket* av_packet,
                    AVFrame* av_frame);

  void StoreContextMerged(const XMA_CONTEXT_DATA& data,
                          const XMA_CONTEXT_DATA& initial_data,
                          uint8_t* context_ptr);

  // AVPacket payload buffer (xma::BuildAvPacketPayload writes into this).
  std::array<uint8_t, 1 + 4096> xma_frame_;
  // Up to two-channel decoded PCM, written by ConvertFrame().
  std::array<uint8_t, kBytesPerFrameChannel * 2> raw_frame_;

  int32_t remaining_subframe_blocks_in_output_buffer_ = 0;
  uint8_t current_frame_remaining_subframes_ = 0;

  uint8_t loop_frame_output_limit_ = 0;
  bool loop_start_skip_pending_ = false;

  // Per-decode snapshot — useful when reading a captured manifest /
  // reproducing cutscene corruption.
  uint64_t decode_attempt_count_ = 0;
  uint32_t last_input_read_offset_before_ = 0;
  uint32_t last_input_read_offset_after_ = 0;
  uint32_t last_current_input_packet_count_ = 0;
  uint32_t last_frame_size_bits_ = 0;
  uint32_t last_bits_to_copy_ = 0;
  uint32_t last_next_packet_index_ = 0;
  uint8_t last_current_buffer_ = 0;
  uint8_t last_skip_count_ = 0;
  int32_t last_packet_index_ = -1;
  bool last_cross_packet_copy_ = false;
  bool last_swapped_input_buffer_ = false;
  bool last_decode_succeeded_ = false;
  uint32_t last_error_status_ = 0;
};

}  // namespace apu
}  // namespace xe

#endif  // XENIA_APU_XMA_CONTEXT_V3_H_
