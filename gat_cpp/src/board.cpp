#include "board.hpp"

#include <algorithm>
#include <stdexcept>

Board::Board(int width, int height, int n_in_row)
    : width_(width),
      height_(height),
      n_in_row_(n_in_row),
      players_({1, 2}),
      current_player_(1),
      last_move_(-1) {}

void Board::init_board(int start_player) {
    if (width_ < n_in_row_ || height_ < n_in_row_) {
        throw std::runtime_error("board size must be >= n_in_row");
    }
    if (start_player != 0 && start_player != 1) {
        throw std::runtime_error("start_player must be 0 or 1");
    }

    current_player_ = players_[start_player];
    availables_.clear();
    availables_.reserve(width_ * height_);
    for (int i = 0; i < width_ * height_; ++i) {
        availables_.push_back(i);
    }
    states_.clear();
    last_move_ = -1;
}

std::pair<int, int> Board::move_to_location(int move) const {
    return {move / width_, move % width_};
}

int Board::location_to_move(int row, int col) const {
    if (row < 0 || row >= height_ || col < 0 || col >= width_) {
        return -1;
    }
    return row * width_ + col;
}

void Board::do_move(int move) {
    auto it = std::find(availables_.begin(), availables_.end(), move);
    if (it == availables_.end()) {
        throw std::runtime_error("illegal move");
    }

    states_[move] = current_player_;
    availables_.erase(it);
    current_player_ = (current_player_ == players_[0]) ? players_[1] : players_[0];
    last_move_ = move;
}

std::pair<bool, int> Board::has_a_winner() const {
    if (last_move_ < 0 || static_cast<int>(states_.size()) < n_in_row_) {
        return {false, -1};
    }

    const int p = states_.at(last_move_);
    const int h = last_move_ / width_;
    const int w = last_move_ % width_;

    const int dirs[4][2] = {
        {0, 1}, {1, 0}, {1, 1}, {1, -1}
    };

    for (const auto& d : dirs) {
        int count = 1;
        for (int sign : {-1, 1}) {
            for (int step = 1; step < n_in_row_; ++step) {
                const int nh = h + sign * step * d[0];
                const int nw = w + sign * step * d[1];
                if (nh < 0 || nh >= height_ || nw < 0 || nw >= width_) {
                    break;
                }
                const int nm = nh * width_ + nw;
                auto it = states_.find(nm);
                if (it != states_.end() && it->second == p) {
                    ++count;
                } else {
                    break;
                }
            }
        }
        if (count >= n_in_row_) {
            return {true, p};
        }
    }

    return {false, -1};
}

std::pair<bool, int> Board::game_end() const {
    auto [win, winner] = has_a_winner();
    if (win) {
        return {true, winner};
    }
    if (availables_.empty()) {
        return {true, -1};
    }
    return {false, -1};
}
