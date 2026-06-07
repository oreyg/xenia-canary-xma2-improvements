/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_LOC_LOCALIZATION_H_
#define XENIA_LOC_LOCALIZATION_H_

#include <filesystem>

namespace xe {
namespace loc {

// Registers the localization JIT prolog hooks (in-game text extraction, live
// translation, and clipboard export) and loads text_hook_config. Call once
// during emulator setup, before any guest code is translated. A relative
// text_hook_config path is resolved against `config_folder` (the same storage
// root as xenia-canary.config.toml).
void Initialize(const std::filesystem::path& config_folder);

}  // namespace loc
}  // namespace xe

#endif  // XENIA_LOC_LOCALIZATION_H_
