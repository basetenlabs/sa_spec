// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <functional>
#include <memory>

namespace sa_spec {

struct LockReleaseGuard {
    virtual ~LockReleaseGuard() = default;
};

std::unique_ptr<LockReleaseGuard> scoped_gil_release();

void set_lock_release_guard_factory(std::function<std::unique_ptr<LockReleaseGuard>()> factory);

} // namespace sa_spec
