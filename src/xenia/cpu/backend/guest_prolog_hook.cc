/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/guest_prolog_hook.h"

namespace xe {
namespace cpu {
namespace backend {

// Function-local storage so the vector is constructed before any static
// initializer that registers a hook, regardless of link order.
static std::vector<GuestPrologHook>& MutableGuestPrologHooks() {
  static std::vector<GuestPrologHook> hooks;
  return hooks;
}

void RegisterGuestPrologHook(const GuestPrologHook& hook) {
  MutableGuestPrologHooks().push_back(hook);
}

const std::vector<GuestPrologHook>& GuestPrologHooks() {
  return MutableGuestPrologHooks();
}

}  // namespace backend
}  // namespace cpu
}  // namespace xe
