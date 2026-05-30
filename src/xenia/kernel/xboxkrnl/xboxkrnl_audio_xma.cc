/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/audio_system.h"
#include "xenia/apu/xma_decoder.h"
#include "xenia/base/clock.h"
#include "xenia/base/logging.h"
#include "xenia/emulator.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {
namespace xboxkrnl {

using xe::apu::XMA_CONTEXT_DATA;

// See audio_system.cc for implementation details.
//
// XMA details:
// https://devel.nuclex.org/external/svn/directx/trunk/include/xma2defs.h
// https://github.com/gdawg/fsbext/blob/master/src/xma_header.h
//
// XMA is undocumented, but the methods are pretty simple.
// Games do this sequence to decode (now):
//   (not sure we are setting buffer validity/offsets right)
// d> XMACreateContext(20656800)
// d> XMAIsInputBuffer0Valid(000103E0)
// d> XMAIsInputBuffer1Valid(000103E0)
// d> XMADisableContext(000103E0, 0)
// d> XMABlockWhileInUse(000103E0)
// d> XMAInitializeContext(000103E0, 20008810)
// d> XMASetOutputBufferValid(000103E0)
// d> XMASetInputBuffer0Valid(000103E0)
// d> XMAEnableContext(000103E0)
// d> XMAGetOutputBufferWriteOffset(000103E0)
// d> XMAGetOutputBufferReadOffset(000103E0)
// d> XMAIsOutputBufferValid(000103E0)
// d> XMAGetOutputBufferReadOffset(000103E0)
// d> XMAGetOutputBufferWriteOffset(000103E0)
// d> XMAIsInputBuffer0Valid(000103E0)
// d> XMAIsInputBuffer1Valid(000103E0)
// d> XMAIsInputBuffer0Valid(000103E0)
// d> XMAIsInputBuffer1Valid(000103E0)
// d> XMAReleaseContext(000103E0)
//
// XAudio2 uses XMA under the covers, and seems to map with the same
// restrictions of frame/subframe/etc:
// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.xaudio2.xaudio2_buffer(v=vs.85).aspx

// Guests virtual range could be read-write protected:
// Resolve through unprotected alias.
static inline uint8_t* XmaContextHost(uint32_t guest_addr) {
  return kernel_state()
      ->emulator()
      ->audio_system()
      ->xma_decoder()
      ->GetContextDataHostPtr(guest_addr);
}

dword_result_t XMACreateContext_entry(lpdword_t context_out_ptr) {
  auto xma_decoder = kernel_state()->emulator()->audio_system()->xma_decoder();
  uint32_t context_ptr = xma_decoder->AllocateContext();
  *context_out_ptr = context_ptr;
  XELOGAPU("[{}ms] XMA kernel: CreateContext = {:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr);
  if (!context_ptr) {
    return X_STATUS_NO_MEMORY;
  }
  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(XMACreateContext, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMAReleaseContext_entry(lpvoid_t context_ptr) {
  XELOGAPU("[{}ms] XMA kernel: ReleaseContext ctx_ptr={:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address());
  auto xma_decoder = kernel_state()->emulator()->audio_system()->xma_decoder();
  xma_decoder->ReleaseContext(context_ptr);
  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMAReleaseContext, kAudio, kImplemented,
                         kHighFrequency);

void StoreXmaContextIndexedRegister(KernelState* kernel_state,
                                    uint32_t base_reg, uint32_t context_ptr) {
  uint32_t context_physical_address =
      kernel_memory()->GetPhysicalAddress(context_ptr);
  assert_true(context_physical_address != UINT32_MAX);
  auto xma_decoder = kernel_state->emulator()->audio_system()->xma_decoder();
  uint32_t hw_index =
      (context_physical_address - xma_decoder->context_array_ptr()) /
      sizeof(XMA_CONTEXT_DATA);
  uint32_t reg_num = base_reg + (hw_index >> 5) * 4;
  uint32_t reg_value = 1 << (hw_index & 0x1F);
  xma_decoder->WriteRegister(reg_num, xe::byte_swap(reg_value));
}

struct XMA_LOOP_DATA {
  xe::be<uint32_t> loop_start;
  xe::be<uint32_t> loop_end;
  uint8_t loop_count;
  uint8_t loop_subframe_end;
  uint8_t loop_subframe_skip;
};
static_assert_size(XMA_LOOP_DATA, 12);

struct XMA_CONTEXT_INIT {
  xe::be<uint32_t> input_buffer_0_ptr;
  xe::be<uint32_t> input_buffer_0_packet_count;
  xe::be<uint32_t> input_buffer_1_ptr;
  xe::be<uint32_t> input_buffer_1_packet_count;
  xe::be<uint32_t> input_buffer_read_offset;
  xe::be<uint32_t> output_buffer_ptr;
  xe::be<uint32_t> output_buffer_block_count;
  xe::be<uint32_t> work_buffer;
  xe::be<uint32_t> subframe_decode_count;
  xe::be<uint32_t> channel_count;
  xe::be<uint32_t> sample_rate;
  XMA_LOOP_DATA loop_data;
};
static_assert_size(XMA_CONTEXT_INIT, 56);

dword_result_t XMAInitializeContext_entry(
    lpvoid_t context_ptr, pointer_t<XMA_CONTEXT_INIT> context_init) {
  XELOGAPU("[{}ms] XMA kernel: InitializeContext ctx_ptr={:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address());
  // Input buffers may be null (buffer 1 in 415607D4).
  // Convert to host endianness.
  uint32_t input_buffer_0_guest_ptr = context_init->input_buffer_0_ptr;
  uint32_t input_buffer_0_physical_address = 0;
  if (input_buffer_0_guest_ptr) {
    input_buffer_0_physical_address =
        kernel_memory()->GetPhysicalAddress(input_buffer_0_guest_ptr);
    // Xenia-specific safety check.
    assert_true(input_buffer_0_physical_address != UINT32_MAX);
    if (input_buffer_0_physical_address == UINT32_MAX) {
      XELOGE(
          "XMAInitializeContext: Invalid input buffer 0 virtual address {:08X}",
          input_buffer_0_guest_ptr);
      return X_E_FALSE;
    }
  }
  uint32_t input_buffer_1_guest_ptr = context_init->input_buffer_1_ptr;
  uint32_t input_buffer_1_physical_address = 0;
  if (input_buffer_1_guest_ptr) {
    input_buffer_1_physical_address =
        kernel_memory()->GetPhysicalAddress(input_buffer_1_guest_ptr);
    assert_true(input_buffer_1_physical_address != UINT32_MAX);
    if (input_buffer_1_physical_address == UINT32_MAX) {
      XELOGE(
          "XMAInitializeContext: Invalid input buffer 1 virtual address {:08X}",
          input_buffer_1_guest_ptr);
      return X_E_FALSE;
    }
  }
  uint32_t output_buffer_guest_ptr = context_init->output_buffer_ptr;
  assert_not_zero(output_buffer_guest_ptr);
  uint32_t output_buffer_physical_address =
      kernel_memory()->GetPhysicalAddress(output_buffer_guest_ptr);
  assert_true(output_buffer_physical_address != UINT32_MAX);
  if (output_buffer_physical_address == UINT32_MAX) {
    XELOGE("XMAInitializeContext: Invalid output buffer virtual address {:08X}",
           output_buffer_guest_ptr);
    return X_E_FALSE;
  }

  uint8_t* host_ctx = XmaContextHost(context_ptr.guest_address());
  std::memset(host_ctx, 0, sizeof(XMA_CONTEXT_DATA));

  XMA_CONTEXT_DATA context(host_ctx);

  context.input_buffer_0_ptr = input_buffer_0_physical_address;
  context.input_buffer_0_packet_count =
      context_init->input_buffer_0_packet_count;
  context.input_buffer_1_ptr = input_buffer_1_physical_address;
  context.input_buffer_1_packet_count =
      context_init->input_buffer_1_packet_count;
  context.input_buffer_read_offset = context_init->input_buffer_read_offset;
  context.output_buffer_ptr = output_buffer_physical_address;
  context.output_buffer_block_count = context_init->output_buffer_block_count;

  // context.work_buffer = context_init->work_buffer;  // ?
  context.subframe_decode_count = context_init->subframe_decode_count;
  context.is_stereo = context_init->channel_count >= 1;
  context.sample_rate = context_init->sample_rate;

  context.loop_start = context_init->loop_data.loop_start;
  context.loop_end = context_init->loop_data.loop_end;
  context.loop_count = context_init->loop_data.loop_count;
  context.loop_subframe_end = context_init->loop_data.loop_subframe_end;
  context.loop_subframe_skip = context_init->loop_data.loop_subframe_skip;

  context.Store(XmaContextHost(context_ptr.guest_address()));

  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMAInitializeContext, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMASetLoopData_entry(lpvoid_t context_ptr,
                                    pointer_t<XMA_LOOP_DATA> loop_data) {
  XELOGAPU("[{}ms] XMA kernel: SetLoopData ctx_ptr={:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address());
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));

  context.loop_start = loop_data->loop_start;
  context.loop_end = loop_data->loop_end;
  context.loop_count = loop_data->loop_count;
  context.loop_subframe_end = loop_data->loop_subframe_end;
  context.loop_subframe_skip = loop_data->loop_subframe_skip;

  context.Store(XmaContextHost(context_ptr.guest_address()));

  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMASetLoopData, kAudio, kImplemented, kHighFrequency);

dword_result_t XMAGetInputBufferReadOffset_entry(lpvoid_t context_ptr) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU(
      "[{}ms] XMA kernel: GetInputBufferReadOffset ctx_ptr={:08X} value={}",
      Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
      static_cast<uint32_t>(context.input_buffer_read_offset));
  return context.input_buffer_read_offset;
}
DECLARE_XBOXKRNL_EXPORT2(XMAGetInputBufferReadOffset, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMASetInputBufferReadOffset_entry(lpvoid_t context_ptr,
                                                 dword_t value) {
  XELOGAPU(
      "[{}ms] XMA kernel: SetInputBufferReadOffset ctx_ptr={:08X} value={}",
      Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
      static_cast<uint32_t>(value));
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  context.input_buffer_read_offset = value;
  context.Store(XmaContextHost(context_ptr.guest_address()));

  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMASetInputBufferReadOffset, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMASetInputBuffer0_entry(lpvoid_t context_ptr, lpvoid_t buffer,
                                        dword_t packet_count) {
  XELOGAPU(
      "[{}ms] XMA kernel: SetInputBuffer0 ctx_ptr={:08X} buffer={:08X} pkts={}",
      Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
      buffer.guest_address(), static_cast<uint32_t>(packet_count));
  uint32_t buffer_physical_address =
      kernel_memory()->GetPhysicalAddress(buffer.guest_address());
  assert_true(buffer_physical_address != UINT32_MAX);
  if (buffer_physical_address == UINT32_MAX) {
    // Xenia-specific safety check.
    XELOGE("XMASetInputBuffer0: Invalid buffer virtual address {:08X}",
           buffer.guest_address());
    return X_E_FALSE;
  }

  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));

  context.input_buffer_0_ptr = buffer_physical_address;
  context.input_buffer_0_packet_count = packet_count;

  context.Store(XmaContextHost(context_ptr.guest_address()));

  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMASetInputBuffer0, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMAIsInputBuffer0Valid_entry(lpvoid_t context_ptr) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU("[{}ms] XMA kernel: IsInputBuffer0Valid ctx_ptr={:08X} value={}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
           static_cast<uint32_t>(context.input_buffer_0_valid));
  return context.input_buffer_0_valid;
}
DECLARE_XBOXKRNL_EXPORT2(XMAIsInputBuffer0Valid, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMASetInputBuffer0Valid_entry(lpvoid_t context_ptr) {
  XELOGAPU("[{}ms] XMA kernel: SetInputBuffer0Valid ctx_ptr={:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address());
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  context.input_buffer_0_valid = 1;
  context.Store(XmaContextHost(context_ptr.guest_address()));

  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMASetInputBuffer0Valid, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMASetInputBuffer1_entry(lpvoid_t context_ptr, lpvoid_t buffer,
                                        dword_t packet_count) {
  XELOGAPU(
      "[{}ms] XMA kernel: SetInputBuffer1 ctx_ptr={:08X} buffer={:08X} pkts={}",
      Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
      buffer.guest_address(), static_cast<uint32_t>(packet_count));
  uint32_t buffer_physical_address =
      kernel_memory()->GetPhysicalAddress(buffer.guest_address());
  assert_true(buffer_physical_address != UINT32_MAX);
  if (buffer_physical_address == UINT32_MAX) {
    // Xenia-specific safety check.
    XELOGE("XMASetInputBuffer1: Invalid buffer virtual address {:08X}",
           buffer.guest_address());
    return X_E_FALSE;
  }

  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));

  context.input_buffer_1_ptr = buffer_physical_address;
  context.input_buffer_1_packet_count = packet_count;

  context.Store(XmaContextHost(context_ptr.guest_address()));

  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMASetInputBuffer1, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMAIsInputBuffer1Valid_entry(lpvoid_t context_ptr) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU("[{}ms] XMA kernel: IsInputBuffer1Valid ctx_ptr={:08X} value={}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
           static_cast<uint32_t>(context.output_buffer_valid));
  return context.input_buffer_1_valid;
}
DECLARE_XBOXKRNL_EXPORT2(XMAIsInputBuffer1Valid, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMASetInputBuffer1Valid_entry(lpvoid_t context_ptr) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  context.input_buffer_1_valid = 1;
  context.Store(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU("[{}ms] XMA kernel: SetInputBuffer1Valid ctx_ptr={:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address());
  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMASetInputBuffer1Valid, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMAIsOutputBufferValid_entry(lpvoid_t context_ptr) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU("[{}ms] XMA kernel: IsOutputBufferValid ctx_ptr={:08X} value={}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
           static_cast<uint32_t>(context.output_buffer_valid));
  return context.output_buffer_valid;
}
DECLARE_XBOXKRNL_EXPORT2(XMAIsOutputBufferValid, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMASetOutputBufferValid_entry(lpvoid_t context_ptr) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  context.output_buffer_valid = 1;
  context.Store(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU("[{}ms] XMA kernel: SetOutputBufferValid ctx_ptr={:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address());
  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMASetOutputBufferValid, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMAGetOutputBufferReadOffset_entry(lpvoid_t context_ptr) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU(
      "[{}ms] XMA kernel: GetOutputBufferReadOffset ctx_ptr={:08X} value={}",
      Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
      static_cast<uint32_t>(context.output_buffer_read_offset));
  return context.output_buffer_read_offset;
}
DECLARE_XBOXKRNL_EXPORT2(XMAGetOutputBufferReadOffset, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMASetOutputBufferReadOffset_entry(lpvoid_t context_ptr,
                                                  dword_t value) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  context.output_buffer_read_offset = value;
  context.Store(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU(
      "[{}ms] XMA kernel: SetOutputBufferReadOffset ctx_ptr={:08X} value={}",
      Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
      static_cast<uint32_t>(value));
  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMASetOutputBufferReadOffset, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMAGetOutputBufferWriteOffset_entry(lpvoid_t context_ptr) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU(
      "[{}ms] XMA kernel: GetOutputBufferWriteOffset ctx_ptr={:08X} "
      "value={}",
      Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
      static_cast<uint32_t>(context.output_buffer_write_offset));
  return context.output_buffer_write_offset;
}
DECLARE_XBOXKRNL_EXPORT2(XMAGetOutputBufferWriteOffset, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMAGetPacketMetadata_entry(lpvoid_t context_ptr) {
  std::lock_guard lock(xe::apu::XmaContext::global_lock_);
  XMA_CONTEXT_DATA context(XmaContextHost(context_ptr.guest_address()));
  XELOGAPU("[{}ms] XMA kernel: GetPacketMetadata ctx_ptr={:08X} value={}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
           static_cast<uint32_t>(context.packet_metadata));
  return context.packet_metadata;
}
DECLARE_XBOXKRNL_EXPORT1(XMAGetPacketMetadata, kAudio, kImplemented);

dword_result_t XMAEnableContext_entry(lpvoid_t context_ptr) {
  XELOGAPU("[{}ms] XMA kernel: EnableContext ctx_ptr={:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address());
  StoreXmaContextIndexedRegister(kernel_state(), 0x1A80, context_ptr);
  StoreXmaContextIndexedRegister(kernel_state(), 0x1940, context_ptr);
  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMAEnableContext, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMADisableContext_entry(lpvoid_t context_ptr, dword_t wait) {
  X_HRESULT result = X_E_SUCCESS;
  XELOGAPU("[{}ms] XMA kernel: DisableContext ctx_ptr={:08X} wait={}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address(),
           static_cast<uint32_t>(wait));
  StoreXmaContextIndexedRegister(kernel_state(), 0x1A40, context_ptr);
  if (!kernel_state()
           ->emulator()
           ->audio_system()
           ->xma_decoder()
           ->BlockOnContext(context_ptr, !wait)) {
    result = X_E_FALSE;
  }
  return result;
}
DECLARE_XBOXKRNL_EXPORT2(XMADisableContext, kAudio, kImplemented,
                         kHighFrequency);

dword_result_t XMABlockWhileInUse_entry(lpvoid_t context_ptr) {
  XELOGAPU("[{}ms] XMA kernel: BlockWhileInUse enter ctx_ptr={:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address());
  // The XMA worker holds global_lock_ for the duration of its batch. Spin
  // on try_lock so the kernel call waits until decoder isn't actively
  // touching contexts.
  while (true) {
    std::unique_lock lock(xe::apu::XmaContext::global_lock_, std::try_to_lock);
    if (lock.owns_lock()) {
      break;
    }
    xe::threading::MaybeYield();
  }
  XELOGAPU("[{}ms] XMA kernel: BlockWhileInUse done ctx_ptr={:08X}",
           Clock::QueryHostUptimeMillis(), context_ptr.guest_address());
  return 0;
}
DECLARE_XBOXKRNL_EXPORT2(XMABlockWhileInUse, kAudio, kImplemented,
                         kHighFrequency);

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

DECLARE_XBOXKRNL_EMPTY_REGISTER_EXPORTS(AudioXma);
