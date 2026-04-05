#include "board.hpp"
#include "mcts_stub.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
        .def("get_move_probs", &AlphaZeroPlayer::get_move_probs, py::arg("board"), py::arg("temp") = 1e-3F, py::call_guard<py::gil_scoped_release>())
        .def("get_action", &AlphaZeroPlayer::get_action, py::call_guard<py::gil_scoped_release>());
}
