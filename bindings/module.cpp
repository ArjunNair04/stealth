#include "stealth/feed_forward.hpp"
#include "stealth/runtime.hpp"
#include "stealth/model_file.hpp"
#include "stealth/window_stream.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>
#include <string>

namespace py = pybind11;
void bind_signal_index(py::module_ &m);
using namespace stealth::numeric;
namespace {
py::array array(const py::object &object, int dimensions) {
    if (!py::isinstance<py::array>(object))
        throw py::type_error("Expected a NumPy array");
    auto a = py::reinterpret_borrow<py::array>(object);
    if (!a.dtype().equal(py::dtype::of<float>()))
        throw py::type_error("Expected native-endian float32");
    if (a.ndim() != dimensions || !(a.flags() & py::array::c_style))
        throw py::value_error("Expected a C-contiguous array with the documented dimensions");
    if (reinterpret_cast<std::uintptr_t>(a.data()) % alignof(float) != 0)
        throw py::value_error("Expected aligned float32 storage");
    if (a.size() <= 0 || static_cast<std::size_t>(a.size()) > max_elements)
        throw py::value_error("Array is empty or exceeds the element limit");
    return a;
}
std::span<const float> values(const py::array &a) {
    return {static_cast<const float *>(a.data()), static_cast<std::size_t>(a.size())};
}
std::span<float> values(py::array_t<float> &a) {
    return {a.mutable_data(), static_cast<std::size_t>(a.size())};
}
void check(Error e) {
    if (e == Error::ok)
        return;
    if (e == Error::invalid_state)
        throw std::runtime_error(error_name(e));
    if (e == Error::numerical_error)
        throw std::overflow_error(error_name(e));
    throw py::value_error(error_name(e));
}
DenseKernel kernel(const std::string &name) {
    if (name == "packed")
        return DenseKernel::packed;
    if (name == "reference")
        return DenseKernel::reference;
    throw py::value_error("kernel must be packed or reference");
}
struct BusyGuard {
    std::atomic<bool> &busy;
    explicit BusyGuard(std::atomic<bool> &flag) : busy(flag) {
        if (busy.exchange(true, std::memory_order_acquire))
            throw std::runtime_error(
                "Session is busy; use a separate session per concurrent caller");
    }
    ~BusyGuard() { busy.store(false, std::memory_order_release); }
    BusyGuard(const BusyGuard &) = delete;
    BusyGuard &operator=(const BusyGuard &) = delete;
};
struct Model {
    std::shared_ptr<const ForecastModel> core;
};
struct Session {
    std::shared_ptr<const ForecastModel> model;
    ForecastWorkspace workspace;
    AlignedBuffer input;
    std::atomic<bool> busy{false};
    explicit Session(const Model &m)
        : model(m.core), workspace(model->config()),
          input(model->config().context * model->config().features) {}
    py::array_t<float> predict(const py::object &object) {
        BusyGuard guard(busy);
        auto a = array(object, 2);
        const auto c = model->config();
        if (a.shape(1) != static_cast<py::ssize_t>(c.features) ||
            a.shape(0) > static_cast<py::ssize_t>(c.context))
            throw py::value_error("Expected [tokens,features] within model context");
        const auto tokens = static_cast<std::size_t>(a.shape(0));
        std::copy(values(a).begin(), values(a).end(), input.span().begin());
        py::array_t<float> output(a.shape(0));
        const auto out = values(output);
        Error status;
        {
            py::gil_scoped_release release;
            status =
                model->forward(input.span().first(tokens * c.features), tokens, out, workspace);
        }
        check(status);
        return output;
    }
    float predict_last(const py::object &object) {
        BusyGuard guard(busy);
        auto a = array(object, 2);
        const auto c = model->config();
        if (a.shape(1) != static_cast<py::ssize_t>(c.features) ||
            a.shape(0) > static_cast<py::ssize_t>(c.context))
            throw py::value_error("Expected [tokens,features] within model context");
        const auto tokens = static_cast<std::size_t>(a.shape(0));
        std::copy(values(a).begin(), values(a).end(), input.span().begin());
        float output = 0;
        Error status;
        {
            py::gil_scoped_release release;
            status = model->forward(input.span().first(tokens * c.features), tokens,
                                    std::span<float>(&output, 1), workspace, SequenceOutput::last);
        }
        check(status);
        return output;
    }
    py::array_t<float> predict_batch(const py::object &object) {
        return batch(object, SequenceOutput::all);
    }
    py::array_t<float> predict_last_batch(const py::object &object) {
        return batch(object, SequenceOutput::last);
    }
    py::array_t<float> batch(const py::object &object, SequenceOutput mode) {
        BusyGuard guard(busy);
        auto a = array(object, 3);
        const auto c = model->config();
        if (a.shape(2) != static_cast<py::ssize_t>(c.features) ||
            a.shape(1) > static_cast<py::ssize_t>(c.context))
            throw py::value_error("Expected [batch,tokens,features] within model context");
        const auto tokens = static_cast<std::size_t>(a.shape(1));
        const auto batch = static_cast<std::size_t>(a.shape(0));
        const auto count = static_cast<std::size_t>(a.size());
        const auto output_tokens = mode == SequenceOutput::last ? std::size_t{1} : tokens;
        if (batch * output_tokens > max_elements - count)
            throw py::value_error("Combined batch input/output exceeds element limit");
        AlignedBuffer snapshot(count);
        std::copy(values(a).begin(), values(a).end(), snapshot.span().begin());
        py::array_t<float> output = mode == SequenceOutput::last
                                        ? py::array_t<float>(a.shape(0))
                                        : py::array_t<float>({a.shape(0), a.shape(1)});
        auto out = values(output);
        Error status = Error::ok;
        {
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < batch; ++i) {
                status = model->forward(
                    snapshot.span().subspan(i * tokens * c.features, tokens * c.features), tokens,
                    out.subspan(i * output_tokens, output_tokens), workspace, mode);
                if (status != Error::ok)
                    break;
            }
        }
        check(status);
        return output;
    }
};
struct Stream {
    std::shared_ptr<const ForecastModel> model;
    WindowStream core;
    AlignedBuffer input;
    std::atomic<bool> busy{false};
    explicit Stream(const Model &m) : model(m.core), core(model), input(model->config().features) {}
    py::object push(const py::object &object) {
        BusyGuard guard(busy);
        auto a = array(object, 1);
        if (values(a).size() != input.size())
            throw py::value_error("Expected one complete feature row");
        std::copy(values(a).begin(), values(a).end(), input.span().begin());
        float output = 0;
        Error status;
        {
            py::gil_scoped_release release;
            status = core.push(input.span(), std::span<float>(&output, 1));
        }
        check(status);
        if (!core.ready())
            return py::none();
        return py::float_(output);
    }
};
} // namespace
PYBIND11_MODULE(_core, m) {
    m.doc() = "Checked native CPU inference. No implicit casts or autograd.";
    m.def("build_info", [] {
        const auto info = stealth::build_info();
        py::dict result;
        result["version"] = std::string(info.version);
        result["compiler"] = std::string(info.compiler);
        result["operating_system"] = std::string(info.operating_system);
        result["architecture"] = std::string(info.architecture);
        result["configuration"] = std::string(info.configuration);
        result["source_fingerprint"] = std::string(info.source_fingerprint);
        return result;
    });
    m.def("gelu", [](const py::object &x) {
        auto a = array(x, 1);
        py::array_t<float> y(a.size());
        AlignedBuffer scratch(static_cast<std::size_t>(a.size()));
        check(gelu(values(a), values(y), scratch.span()));
        return y;
    });
    m.def(
        "softmax",
        [](const py::object &x, const py::object &active) {
            auto a = array(x, 1);
            const auto n =
                active.is_none() ? static_cast<std::size_t>(a.size()) : active.cast<std::size_t>();
            py::array_t<float> y(a.size());
            AlignedBuffer scratch(static_cast<std::size_t>(a.size()));
            check(softmax(values(a), n, values(y), scratch.span()));
            return y;
        },
        py::arg("x"), py::arg("active") = py::none());
    m.def(
        "layer_norm",
        [](const py::object &x, const py::object &scale, const py::object &shift, float epsilon) {
            auto a = array(x, 1), g = array(scale, 1), b = array(shift, 1);
            py::array_t<float> y(a.size());
            AlignedBuffer scratch(static_cast<std::size_t>(a.size()));
            check(layer_norm(values(a), values(g), values(b), epsilon, values(y), scratch.span()));
            return y;
        },
        py::arg("x"), py::arg("scale"), py::arg("shift"), py::arg("epsilon") = 1e-5F);
    m.def("residual_add", [](const py::object &x, const py::object &residual) {
        auto a = array(x, 1), b = array(residual, 1);
        py::array_t<float> y(a.size());
        AlignedBuffer scratch(static_cast<std::size_t>(a.size()));
        check(residual_add(values(a), values(b), values(y), scratch.span()));
        return y;
    });
    py::class_<Dense>(m, "Dense")
        .def(py::init([](const py::object &weights, const py::object &bias, const std::string &k) {
                 auto w = array(weights, 2), b = array(bias, 1);
                 return std::make_unique<Dense>(static_cast<std::size_t>(w.shape(1)),
                                                static_cast<std::size_t>(w.shape(0)), values(w),
                                                values(b), kernel(k));
             }),
             py::arg("weights"), py::arg("bias"), py::arg("kernel") = "packed")
        .def("forward",
             [](const Dense &d, const py::object &x) {
                 auto a = array(x, 1);
                 py::array_t<float> y(static_cast<py::ssize_t>(d.outputs()));
                 AlignedBuffer scratch(d.outputs());
                 check(d.forward(values(a), values(y), scratch.span()));
                 return y;
             })
        .def_property_readonly("inputs", &Dense::inputs)
        .def_property_readonly("outputs", &Dense::outputs);
    py::class_<FeedForward>(m, "FeedForward")
        .def(py::init([](const py::object &scale, const py::object &shift,
                         const py::object &up_weights, const py::object &up_bias,
                         const py::object &down_weights, const py::object &down_bias, float epsilon,
                         const std::string &k) {
                 auto g = array(scale, 1), b = array(shift, 1), up = array(up_weights, 2),
                      ub = array(up_bias, 1), down = array(down_weights, 2),
                      db = array(down_bias, 1);
                 if (up.shape(1) != g.size() || down.shape(0) != g.size() ||
                     down.shape(1) != up.shape(0))
                     throw py::value_error("Invalid feed-forward matrix shapes");
                 return std::make_unique<FeedForward>(
                     static_cast<std::size_t>(g.size()), static_cast<std::size_t>(up.shape(0)),
                     FeedForwardParameters{values(g), values(b), values(up), values(ub),
                                           values(down), values(db)},
                     epsilon, kernel(k));
             }),
             py::arg("scale"), py::arg("shift"), py::arg("up_weights"), py::arg("up_bias"),
             py::arg("down_weights"), py::arg("down_bias"), py::arg("epsilon") = 1e-5F,
             py::arg("kernel") = "packed")
        .def("forward", [](const FeedForward &f, const py::object &x) {
            auto a = array(x, 1);
            py::array_t<float> y(a.size());
            FeedForwardWorkspace workspace(f.width(), f.hidden());
            check(f.forward(values(a), values(y), workspace));
            return y;
        });
    m.def(
        "causal_attention",
        [](const py::object &query, const py::object &key, const py::object &value,
           std::size_t heads) {
            auto q = array(query, 2), k = array(key, 2), v = array(value, 2);
            if (q.shape(0) != k.shape(0) || q.shape(1) != k.shape(1) || q.shape(0) != v.shape(0) ||
                q.shape(1) != v.shape(1))
                throw py::value_error("Attention arrays must have equal shapes");
            const auto tokens = static_cast<std::size_t>(q.shape(0));
            AttentionWorkspace workspace(tokens, static_cast<std::size_t>(q.shape(1)), heads);
            py::array_t<float> output({q.shape(0), q.shape(1)});
            check(causal_attention(values(q), values(k), values(v), tokens, values(output),
                                   workspace));
            return output;
        },
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("heads"));
    py::class_<ForecastConfig>(m, "ForecastConfig")
        .def(py::init([](std::size_t features, std::size_t width, std::size_t heads,
                         std::size_t hidden, std::size_t context, std::size_t horizon,
                         std::size_t target, float epsilon, const std::string &output) {
                 if (output != "level" && output != "correction")
                     throw py::value_error("Unknown forecast output mode");
                 ForecastConfig c{
                     features,
                     width,
                     heads,
                     hidden,
                     context,
                     horizon,
                     target,
                     epsilon,
                     output == "correction" ? ForecastOutput::correction : ForecastOutput::level};
                 (void)forecast_parameter_count(c);
                 return c;
             }),
             py::arg("features"), py::arg("width"), py::arg("heads"), py::arg("hidden"),
             py::arg("context"), py::arg("horizon"), py::arg("target"), py::arg("epsilon") = 1e-5F,
             py::arg("output") = "level")
        .def_readonly("features", &ForecastConfig::features)
        .def_readonly("width", &ForecastConfig::width)
        .def_readonly("heads", &ForecastConfig::heads)
        .def_readonly("hidden", &ForecastConfig::hidden)
        .def_readonly("context", &ForecastConfig::context)
        .def_readonly("horizon", &ForecastConfig::horizon)
        .def_readonly("target", &ForecastConfig::target)
        .def_readonly("epsilon", &ForecastConfig::epsilon)
        .def_property_readonly("output", [](ForecastConfig c) {
            return c.output == ForecastOutput::correction ? "correction" : "level";
        });
    m.def("parameter_count", &forecast_parameter_count);
    py::class_<Model>(m, "ForecastModel")
        .def(py::init([](ForecastConfig c, const py::object &p, const std::string &k) {
                 auto a = array(p, 1);
                 return Model{std::make_shared<const ForecastModel>(c, values(a), kernel(k))};
             }),
             py::arg("config"), py::arg("parameters"), py::arg("kernel") = "packed")
        .def_static(
            "from_bytes",
            [](const py::bytes &bytes, const std::string &k) {
                if (py::len(bytes) > max_model_file_bytes)
                    throw py::value_error("Model file exceeds size limit");
                const std::string content = bytes;
                return Model{load_forecaster(
                    std::as_bytes(std::span(content.data(), content.size())), kernel(k))};
            },
            py::arg("artifact"), py::arg("kernel") = "packed")
        .def_static(
            "load",
            [](const std::filesystem::path &path, const std::string &k) {
                return Model{load_forecaster_file(path, kernel(k))};
            },
            py::arg("path"), py::arg("kernel") = "packed")
        .def_property_readonly("config", [](const Model &m) { return m.core->config(); })
        .def_property_readonly("parameter_bytes",
                               [](const Model &m) { return m.core->parameter_bytes(); })
        .def("session", [](const Model &m) { return std::make_unique<Session>(m); })
        .def("window_stream", [](const Model &m) { return std::make_unique<Stream>(m); });
    py::class_<Session>(m, "ForecastSession")
        .def(py::init<const Model &>())
        .def("predict", &Session::predict)
        .def("predict_batch", &Session::predict_batch)
        .def("predict_last", &Session::predict_last)
        .def("predict_last_batch", &Session::predict_last_batch)
        .def_property_readonly(
            "busy", [](const Session &s) { return s.busy.load(std::memory_order_acquire); })
        .def_property_readonly("workspace_bytes", [](const Session &s) {
            return s.workspace.bytes() + s.input.size() * sizeof(float);
        });
    py::class_<Stream>(m, "WindowStream")
        .def(py::init<const Model &>())
        .def("push", &Stream::push)
        .def("reset",
             [](Stream &s) {
                 BusyGuard guard(s.busy);
                 s.core.reset();
             })
        .def_property_readonly("size",
                               [](Stream &s) {
                                   BusyGuard guard(s.busy);
                                   return s.core.size();
                               })
        .def_property_readonly("ready",
                               [](Stream &s) {
                                   BusyGuard guard(s.busy);
                                   return s.core.ready();
                               })
        .def_property_readonly(
            "busy", [](const Stream &s) { return s.busy.load(std::memory_order_acquire); })
        .def_property_readonly("workspace_bytes", [](const Stream &s) {
            return s.core.bytes() + s.input.size() * sizeof(float);
        });
    bind_signal_index(m);
}
