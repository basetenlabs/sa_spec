// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/pair.h>

#include <sa_spec/api.hpp>
#include <sa_spec/util/config.hpp>
#include <sa_spec/util/lock_release_guard.hpp>

namespace py = nanobind;

namespace sa_spec::bind {

struct PyGil : LockReleaseGuard {
    nanobind::gil_scoped_release m_lck;

    virtual ~PyGil() final = default;
};


void add_request(RequestID::ValueType request_id_, const std::vector<Token::ValueType> tokens_) {
    LOG_TRACE("add_request request_id: %ld, tokens.size(): %ld", request_id_, tokens_.size());

    RequestID request_id(request_id_);
    std::span<const Token> tokens(reinterpret_cast<const Token*>(tokens_.data()), tokens_.size());
    API::get_instance().add_request(request_id, tokens);
}

void prepare(const std::vector<RequestID::ValueType> request_ids_) {
    LOG_TRACE("prepare request_ids.size(): %ld", request_ids_.size());

    std::span<const RequestID> request_ids(reinterpret_cast<const RequestID*>(request_ids_.data()), request_ids_.size());
    API::get_instance().prepare(request_ids);
}


void extend(
    py::ndarray<NumTokens::ValueType, py::ndim<1>, py::c_contig, py::device::cuda> depth_out,
    py::ndarray<Token::ValueType, py::ndim<2>, py::c_contig, py::device::cuda> draft_out,
    py::ndarray<const Token::ValueType, py::ndim<2>, py::c_contig, py::device::cuda> accepted_in,
    py::ndarray<const NumTokens::ValueType, py::ndim<1>, py::c_contig, py::device::cuda> accepted_lens_in) {

    
    [[maybe_unused]] static const auto print_once = [] {
        LOG_INFO("🚀🚀 USING SA_SPEC 🚀🚀");
        return true;
    }();


    int batch_size = static_cast<int>(depth_out.shape(0));
    int draft_length = static_cast<int>(draft_out.shape(1));

    ASSERT_THROW(batch_size == static_cast<int>(draft_out.shape(0)));
    ASSERT_THROW(batch_size == static_cast<int>(accepted_lens_in.shape(0)));
    ASSERT_THROW(draft_length + 1 == static_cast<int>(accepted_in.shape(1)));

    LOG_TRACE("extend batch_size: %d, draft_length: %d", batch_size, draft_length);

    if (batch_size == 0) {
        return;
    }

    API::get_instance().extend(batch_size, draft_length, depth_out.data(), draft_out.data(), accepted_in.data(), accepted_lens_in.data());
}


void get_active_tokens_for_test(
    RequestID::ValueType request_id_,
    py::ndarray<Token::ValueType, py::ndim<1>, py::c_contig, py::device::cuda> tokens_out) {

    Token::ValueType* tokens_out_data = tokens_out.data();
    size_t tokens_out_len = tokens_out.shape(0);

    LOG_TRACE("get_active_tokens_for_test request_id: %ld, tokens_out.size(): %ld", request_id_, tokens_out_len);

    API::get_instance().get_active_tokens_for_test(RequestID(request_id_), tokens_out_data, tokens_out_len);
}


} // namespace sa_spec::bind


NB_MODULE(_sa_spec_impl, m) {
    using namespace py::literals;

    sa_spec::set_lock_release_guard_factory([]() -> std::unique_ptr<sa_spec::LockReleaseGuard> {
        return std::make_unique<sa_spec::bind::PyGil>();
    });

    m.def("add_request", &sa_spec::bind::add_request, "request_id"_a, "tokens"_a);

    m.def("prepare", &sa_spec::bind::prepare, "request_ids"_a);

    m.def("extend", &sa_spec::bind::extend,
          "depth_out"_a.noconvert(),
          "draft_out"_a.noconvert(),
          "accepted_in"_a.noconvert(),
          "accepted_lens_in"_a.noconvert());
 
    m.def("get_active_tokens_for_test",
          &sa_spec::bind::get_active_tokens_for_test,
          "request_id"_a,
          "tokens_out"_a.noconvert());
    

    m.attr("MAX_SLOTS") = py::int_(sa_spec::Config::MAX_SLOTS);
}
