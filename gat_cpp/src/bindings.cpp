#include "board.hpp"
#include "mcts_stub.hpp"

#include <pybind11/functional.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS engine scaffold for gomoku";

    py::class_<Board>(m, "Board")
        .def(py::init<int, int, int>(), py::arg("width") = 8, py::arg("height") = 8, py::arg("n_in_row") = 5)
        .def("init_board", &Board::init_board, py::arg("start_player") = 0)
        .def("move_to_location", &Board::move_to_location)
        .def("location_to_move", &Board::location_to_move)
        .def("do_move", &Board::do_move)
        .def("has_a_winner", &Board::has_a_winner)
        .def("game_end", &Board::game_end)
        .def_property_readonly("width", &Board::width)
        .def_property_readonly("height", &Board::height)
        .def_property_readonly("n_in_row", &Board::n_in_row)
        .def_property_readonly("current_player", &Board::current_player)
        .def_property_readonly("last_move", &Board::last_move)
        .def_property_readonly("availables", &Board::availables)
        .def_property_readonly("states", &Board::states);

    py::class_<EvalResult>(m, "EvalResult")
        .def(py::init<>())
        .def_readwrite("policy", &EvalResult::policy)
        .def_readwrite("value", &EvalResult::value);

    py::class_<AlphaZeroPlayer>(m, "AlphaZeroPlayer")
        .def(
            py::init<int, int, int, int>(),
            py::arg("c_puct") = 5,
            py::arg("n_playout") = 400,
            py::arg("seed") = 42,
            py::arg("num_threads") = 0
        )
        .def("set_player_ind", &AlphaZeroPlayer::set_player_ind)
        .def_property_readonly("player", &AlphaZeroPlayer::player)
        .def_property_readonly("n_playout", &AlphaZeroPlayer::n_playout)
        .def_property_readonly("num_threads", &AlphaZeroPlayer::num_threads)
        .def("set_n_playout", &AlphaZeroPlayer::set_n_playout)
        .def("set_num_threads", &AlphaZeroPlayer::set_num_threads)
        .def("reset_player", &AlphaZeroPlayer::reset_player)
        .def("update_with_move", &AlphaZeroPlayer::update_with_move)
        .def("has_eval_callback", &AlphaZeroPlayer::has_eval_callback)

        // get_move_probs: release GIL so other Python threads can run
        // while C++ does tree traversal.  The eval callback re-acquires
        // the GIL internally when it needs to call the neural network.
        .def("get_move_probs", &AlphaZeroPlayer::get_move_probs,
             py::arg("board"), py::arg("temp") = 1e-3F,
             py::call_guard<py::gil_scoped_release>())
        .def("get_action", &AlphaZeroPlayer::get_action,
             py::call_guard<py::gil_scoped_release>())

        // set_eval_callback:  wrap the user-supplied Python callable so that
        // the GIL is re-acquired each time the C++ search calls back into
        // Python for batch neural-network inference.
        .def("set_eval_callback",
             [](AlphaZeroPlayer& self, py::object py_fn, int batch_size) {
                 // prevent the Python function from being garbage-collected
                 auto py_fn_ptr = std::make_shared<py::object>(std::move(py_fn));

                 BatchEvalFn cpp_fn =
                     [py_fn_ptr](const std::vector<float>& states,
                                 int bs, int w, int h) -> std::vector<EvalResult>
                 {
                     py::gil_scoped_acquire gil;

                     // Build a numpy array that *copies* the state data so
                     // C++ memory lifetime is irrelevant to Python.
                     const auto total = static_cast<py::ssize_t>(states.size());
                     py::array_t<float> np_states({static_cast<py::ssize_t>(bs),
                                                   static_cast<py::ssize_t>(4),
                                                   static_cast<py::ssize_t>(w),
                                                   static_cast<py::ssize_t>(h)});
                     std::memcpy(np_states.mutable_data(), states.data(),
                                 static_cast<size_t>(total) * sizeof(float));

                     // Call the Python function
                     py::object result = (*py_fn_ptr)(np_states, bs, w, h);

                     // Parse result: list of (policy_list, value_float)
                     std::vector<EvalResult> out;
                     out.reserve(static_cast<size_t>(bs));
                     auto result_list = result.cast<py::list>();
                     for (auto& item : result_list) {
                         auto tup = item.cast<py::tuple>();
                         EvalResult er;
                         er.policy = tup[0].cast<std::vector<float>>();
                         er.value  = tup[1].cast<float>();
                         out.push_back(std::move(er));
                     }
                     return out;
                 };

                 self.set_eval_callback(std::move(cpp_fn), batch_size);
             },
             py::arg("fn"), py::arg("batch_size") = 8);
}
