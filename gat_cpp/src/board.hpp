#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

class Board {
public:
    Board(int width = 8, int height = 8, int n_in_row = 5);

    void init_board(int start_player = 0);
    std::pair<int, int> move_to_location(int move) const;
    int location_to_move(int row, int col) const;
    void do_move(int move);
    std::pair<bool, int> has_a_winner() const;
    std::pair<bool, int> game_end() const;

    int width() const { return width_; }
    int height() const { return height_; }
    int n_in_row() const { return n_in_row_; }
    int current_player() const { return current_player_; }
    int last_move() const { return last_move_; }

    const std::vector<int>& availables() const { return availables_; }
    const std::unordered_map<int, int>& states() const { return states_; }

private:
    int width_;
    int height_;
    int n_in_row_;

    std::vector<int> players_;
    int current_player_;
    std::vector<int> availables_;
    std::unordered_map<int, int> states_;
    int last_move_;
};
