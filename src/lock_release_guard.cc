// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <functional>

#include <sa_spec/util/lock_release_guard.hpp>

namespace sa_spec {

static std::function<std::unique_ptr<LockReleaseGuard>()> lock_release_guard_factory = nullptr;
   
void set_lock_release_guard_factory(std::function<std::unique_ptr<LockReleaseGuard>()> factory) {
    lock_release_guard_factory = factory;
}

std::unique_ptr<LockReleaseGuard> scoped_gil_release() {
    if (lock_release_guard_factory == nullptr) {
        return nullptr;
    }
    return lock_release_guard_factory();
}

} // namespace sa_spec
