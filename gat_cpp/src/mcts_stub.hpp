#pragma once

#include "board.hpp"

#include <memory>
#include <mutex>
#include <random>
#include <vector>

class AlphaZeroPlayer {
public:
    AlphaZeroPlayer(int c_puct = 5, int n_playout = 400, int seed = 42, int num_threads = 0);
    ~AlphaZeroPlayer();

    void set_player_ind(int p);
    int player() const { return player_; }
    int n_playout() const { return n_playout_; }
    int num_threads() const { return num_threads_; }

    void set_n_playout(int n_playout);
    void set_num_threads(int num_threads);

    void reset_player();
    void update_with_move(int last_move);

    // Parallel rollout policy: evaluate candidate moves with multi-thread playouts.
    int get_action(const Board& board);
    std::vector<float> get_move_probs(const Board& board, float temp = 1e-3F);

private:
    struct ThreadPoolState;
    struct SearchTreeState;

    int resolve_thread_count() const;
    void ensure_thread_pool(int threads);
    void shutdown_thread_pool();
    void worker_loop(int thread_id);

    int c_puct_;
    int n_playout_;
    int num_threads_;
    int player_;
    std::mt19937 rng_;
    std::mutex action_mutex_;
    std::unique_ptr<ThreadPoolState> pool_state_;
    std::unique_ptr<SearchTreeState> tree_state_;
};
