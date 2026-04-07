#pragma once

#include "board.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

/// Result of evaluating one board position with the neural network.
struct EvalResult {
    std::vector<float> policy;  // board_size elements (prior probabilities)
    float value;                // position value in [-1, 1]
};

/// Batch evaluation callback type.
/// Args: flat state data (batch * 4 * W * H), batch_size, width, height.
/// Returns: vector of EvalResult, one per position in the batch.
using BatchEvalFn = std::function<std::vector<EvalResult>(
    const std::vector<float>&, int, int, int)>;

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

    /// Set neural network batch evaluation callback.
    /// When set, MCTS uses NN policy as priors and NN value instead of rollout.
    /// Search is single-threaded; parallelism comes from multiple workers.
    void set_eval_callback(BatchEvalFn cb, int batch_size = 8);
    bool has_eval_callback() const { return static_cast<bool>(eval_callback_); }

    void reset_player();
    void update_with_move(int last_move);

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
    int eval_batch_size_ = 8;
    std::mt19937 rng_;
    std::mutex action_mutex_;
    std::unique_ptr<ThreadPoolState> pool_state_;
    std::unique_ptr<SearchTreeState> tree_state_;
    BatchEvalFn eval_callback_;
};
