#include "stealth/signal_index.hpp"

#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
using namespace stealth::temporal;
namespace {
template <typename T> T integer(py::handle value) {
    if (!PyLong_CheckExact(value.ptr()))
        throw py::value_error("Expected an exact integer");
    try {
        return py::cast<T>(value);
    } catch (const py::cast_error &) {
        throw py::value_error("Integer outside supported range");
    }
}
py::sequence fields(py::handle value, py::ssize_t count) {
    if (!py::isinstance<py::sequence>(value))
        throw py::value_error("Expected a record sequence");
    auto result = py::reinterpret_borrow<py::sequence>(value);
    if (result.size() != static_cast<py::size_t>(count))
        throw py::value_error("Wrong number of record fields");
    return result;
}
const char *name(SignalState state) {
    switch (state) {
    case SignalState::missing: return "missing";
    case SignalState::withdrawn: return "withdrawn";
    case SignalState::stale: return "stale";
    case SignalState::valid: return "valid";
    }
    throw py::value_error("Invalid signal state");
}
}
void bind_signal_index(py::module_ &m) {
    // Internal numeric boundary; tools/native_signal_timeline.py preserves named schemas.
    py::class_<SignalIndex>(m, "_SignalIndex")
        .def(py::init([](const py::sequence &ages, const py::sequence &rows) {
            if (ages.size() == 0 || ages.size() > max_signal_features ||
                rows.size() > max_signal_deliveries)
                throw py::value_error("Signal index capacity exceeded or empty schema");
            std::vector<SignalLimits> limits;
            limits.reserve(ages.size());
            for (auto value : ages) {
                auto r = fields(value, 2);
                limits.push_back({integer<std::int64_t>(r[0]), integer<std::int64_t>(r[1])});
            }
            std::vector<SignalRecord> records;
            records.reserve(rows.size());
            for (auto value : rows) {
                auto r = fields(value, 7);
                std::optional<double> number;
                if (!r[6].is_none()) {
                    if (!PyFloat_CheckExact(r[6].ptr()) && !PyLong_CheckExact(r[6].ptr()))
                        throw py::value_error("Expected a finite number or None");
                    number = py::cast<double>(r[6]);
                }
                records.push_back({integer<std::size_t>(r[0]), integer<std::uint64_t>(r[1]),
                    integer<std::int64_t>(r[2]), integer<std::int64_t>(r[3]),
                    integer<std::int64_t>(r[4]), integer<std::int64_t>(r[5]), number});
            }
            // All Python inputs have been copied before releasing the GIL.
            py::gil_scoped_release release;
            return SignalIndex(limits, records);
        }))
        .def_property_readonly("record_count", &SignalIndex::record_count)
        .def_property_readonly("index_entry_count", &SignalIndex::index_entry_count)
        .def("snapshot", [](const SignalIndex &index, py::handle clock) {
            const auto decision = integer<std::int64_t>(clock);
            std::vector<SignalFeature> output(index.feature_count());
            SnapshotError error;
            {
                py::gil_scoped_release release;
                error = index.snapshot(decision, output);
            }
            if (error != SnapshotError::ok)
                throw py::value_error("Invalid snapshot decision or output size");
            py::list result;
            for (const auto &f : output) {
                py::object observation = py::none();
                if (f.observation) {
                    const auto &r = *f.observation;
                    py::object value = r.value ? py::object(py::float_(*r.value)) : py::none();
                    observation = py::make_tuple(r.event, r.revision, r.effective_at_ns,
                                                 r.available_at_ns, r.received_at_ns, value);
                }
                result.append(py::make_tuple(name(f.state), observation, f.observation_age_ns,
                                              f.publication_age_ns, f.receipt_age_ns));
            }
            return result;
        });
}
