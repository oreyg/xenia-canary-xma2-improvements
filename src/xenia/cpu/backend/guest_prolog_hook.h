/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_BACKEND_GUEST_PROLOG_HOOK_H_
#define XENIA_CPU_BACKEND_GUEST_PROLOG_HOOK_H_

#include <cstdint>
#include <vector>

namespace xe {
namespace cpu {
namespace backend {

// A native callback emitted (via CallNative) at the entry of guest functions
// that match `should_hook`. This is the generic extension point the backend
// exposes so higher-level modules (e.g. xenia-loc) can instrument guest code
// without the backend depending on them: the module registers a hook at
// startup and the emitter calls it from each matching function's prolog.
struct GuestPrologHook {
  // Evaluated at JIT-compile time (host side) for every guest function being
  // translated. Return true to instrument the function at `guest_address`.
  bool (*should_hook)(uint32_t guest_address);
  // Emitted into the prolog and invoked on the guest thread at function entry.
  // Receives the raw PPCContext* and the guest function address. Its return
  // value is ignored (kept for the CallNative signature).
  uint64_t (*callback)(void* raw_context, uint64_t guest_address);
};

// Registers a prolog hook. Must be called before any guest code is translated
// (e.g. from a one-time Initialize() during emulator setup); the registry is
// read-only afterwards, so reads during JIT need no locking.
void RegisterGuestPrologHook(const GuestPrologHook& hook);

// The registered hooks, in registration order.
const std::vector<GuestPrologHook>& GuestPrologHooks();

}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_GUEST_PROLOG_HOOK_H_
