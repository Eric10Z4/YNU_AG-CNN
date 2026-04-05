#include "mcts_stub.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

struct RootSnapshot {
    int width = 0;
    int height = 0;
    int n_in_row = 0;
    int current_player = 1;
    std::vector<uint8_t> cells;
    std::vector<int> availables;
    std::vector<int> pos;
};

struct RolloutState {
    int width = 0;
    int height = 0;
    int n_in_row = 0;
    int current_player = 1;
    int last_move = -1;
    std::vector<uint8_t> cells;
    std::vector<int> availables;
    std::vector<int> pos;
};

struct Node {
    struct Edge {
        int move = -1;
        std::unique_ptr<Node> child;
    };

    explicit Node(float p = 1.0F) : prior(p) {}

    float prior = 1.0F;
    std::atomic<int> visits{0};
    std::atomic<int> value_sum{0};
    std::atomic<int> virtual_loss{0};

    std::atomic<bool> expanded{false};
    std::mutex expand_mu;
    std::vector<Edge> children;
};

struct RolloutTask {
    std::shared_ptr<RootSnapshot> root;
    Node* root_node = nullptr;
    std::vector<uint64_t> thread_seeds;
    int total_playouts = 0;
    int chunk_size = 64;
    float c_puct = 5.0F;
    std::atomic<int> next_sim{0};
};

bool has_a_winner_fast(const RolloutState& st, int move, int player) {
    if (move < 0 || player <= 0) {
        return false;
    }
    const int h = move / st.width;
    const int w = move % st.width;

    const int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (const auto& d : dirs) {
        int count = 1;
        for (int sign : {-1, 1}) {
            for (int step = 1; step < st.n_in_row; ++step) {
                const int nh = h + sign * step * d[0];
                const int nw = w + sign * step * d[1];
                if (nh < 0 || nh >= st.height || nw < 0 || nw >= st.width) {
                    break;
                }
                const int nm = nh * st.width + nw;
                if (st.cells[static_cast<size_t>(nm)] == static_cast<uint8_t>(player)) {
                    ++count;
                } else {
                    break;
                }
            }
        }
        if (count >= st.n_in_row) {
            return true;
        }
    }
    return false;
}

void do_move_fast(RolloutState& st, int move) {
    const size_t idx = static_cast<size_t>(move);
    if (idx >= st.cells.size()) {
        throw std::runtime_error("illegal move index");
    }
    if (st.cells[idx] != 0U) {
        throw std::runtime_error("illegal move occupied");
    }

    const int pidx = st.pos[idx];
    if (pidx < 0 || static_cast<size_t>(pidx) >= st.availables.size()) {
        throw std::runtime_error("illegal move unavailable");
    }

    st.cells[idx] = static_cast<uint8_t>(st.current_player);
    st.last_move = move;

    const int last = st.availables.back();
    st.availables[static_cast<size_t>(pidx)] = last;
    st.pos[static_cast<size_t>(last)] = pidx;
    st.availables.pop_back();
    st.pos[idx] = -1;

    st.current_player = (st.current_player == 1) ? 2 : 1;
}

RootSnapshot build_root_snapshot(const Board& board) {
    RootSnapshot root;
    root.width = board.width();
    root.height = board.height();
    root.n_in_row = board.n_in_row();
    root.current_player = board.current_player();

    const int board_size = root.width * root.height;
    root.cells.assign(static_cast<size_t>(board_size), 0U);
    root.pos.assign(static_cast<size_t>(board_size), -1);

    for (const auto& kv : board.states()) {
        if (kv.first >= 0 && kv.first < board_size) {
            root.cells[static_cast<size_t>(kv.first)] = static_cast<uint8_t>(kv.second);
        }
    }

    root.availables = board.availables();
    for (size_t i = 0; i < root.availables.size(); ++i) {
        root.pos[static_cast<size_t>(root.availables[i])] = static_cast<int>(i);
    }
    return root;
}

RolloutState make_rollout_state(const RootSnapshot& root) {
    RolloutState st;
    st.width = root.width;
    st.height = root.height;
    st.n_in_row = root.n_in_row;
    st.current_player = root.current_player;
    st.last_move = -1;
    st.cells = root.cells;
    st.availables = root.availables;
    st.pos = root.pos;
    return st;
}

bool evaluate_terminal_for_current_player(const RolloutState& st, int* value) {
    if (st.last_move >= 0) {
        const int just_played = (st.current_player == 1) ? 2 : 1;
        if (has_a_winner_fast(st, st.last_move, just_played)) {
            *value = (just_played == st.current_player) ? 1 : -1;
            return true;
        }
    }
    if (st.availables.empty()) {
        *value = 0;
        return true;
    }
    return false;
}

int rollout_random_from_state(RolloutState st, std::mt19937& rng) {
    const int start_player = st.current_player;
    int terminal_value = 0;
    if (evaluate_terminal_for_current_player(st, &terminal_value)) {
        return terminal_value;
    }

    while (!st.availables.empty()) {
        std::uniform_int_distribution<size_t> pick_idx(0, st.availables.size() - 1);
        const int move = st.availables[pick_idx(rng)];
        do_move_fast(st, move);

        const int just_played = (st.current_player == 1) ? 2 : 1;
        if (has_a_winner_fast(st, st.last_move, just_played)) {
            return (just_played == start_player) ? 1 : -1;
        }
    }
    return 0;
}

void simulate_one(const RootSnapshot& root, Node* root_node, float c_puct, std::mt19937& rng) {
    RolloutState st = make_rollout_state(root);
    Node* node = root_node;
    std::vector<Node*> path;
    std::vector<Node*> virtual_loss_nodes;
    path.reserve(128);
    virtual_loss_nodes.reserve(128);
    path.push_back(node);

    int leaf_value = 0;
    while (true) {
        if (evaluate_terminal_for_current_player(st, &leaf_value)) {
            break;
        }

        bool expanded_now = false;
        if (!node->expanded.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(node->expand_mu);
            if (!node->expanded.load(std::memory_order_relaxed)) {
                const float prior = 1.0F / static_cast<float>(std::max<size_t>(1, st.availables.size()));
                node->children.reserve(st.availables.size());
                for (int mv : st.availables) {
                    Node::Edge e;
                    e.move = mv;
                    e.child = std::make_unique<Node>(prior);
                    node->children.push_back(std::move(e));
                }
                node->expanded.store(true, std::memory_order_release);
                expanded_now = true;
            }
        }

        if (expanded_now) {
            leaf_value = rollout_random_from_state(st, rng);
            break;
        }

        Node* best_child = nullptr;
        int best_move = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        const int parent_visits = std::max(1, node->visits.load(std::memory_order_relaxed));
        const double sqrt_parent = std::sqrt(static_cast<double>(parent_visits));
        for (const auto& edge : node->children) {
            Node* child = edge.child.get();
            const int cv = child->visits.load(std::memory_order_relaxed);
            const int cs = child->value_sum.load(std::memory_order_relaxed);
            const int vl = child->virtual_loss.load(std::memory_order_relaxed);
            const double q = (cv > 0) ? (static_cast<double>(cs) / static_cast<double>(cv)) : 0.0;
            const double u = static_cast<double>(c_puct) * static_cast<double>(child->prior) * sqrt_parent /
                             static_cast<double>(1 + cv);
            const double score = q + u - static_cast<double>(vl);
            if (score > best_score) {
                best_score = score;
                best_move = edge.move;
                best_child = child;
            }
        }

        if (best_child == nullptr || best_move < 0) {
            leaf_value = rollout_random_from_state(st, rng);
            break;
        }

        best_child->virtual_loss.fetch_add(1, std::memory_order_relaxed);
        virtual_loss_nodes.push_back(best_child);
        do_move_fast(st, best_move);
        node = best_child;
        path.push_back(node);
    }

    for (Node* n : virtual_loss_nodes) {
        n->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
    }

    int v = leaf_value;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        (*it)->visits.fetch_add(1, std::memory_order_relaxed);
        (*it)->value_sum.fetch_add(v, std::memory_order_relaxed);
        v = -v;
    }
}

}  // namespace

struct AlphaZeroPlayer::ThreadPoolState {
    int worker_count = 0;
    bool stop = false;
    uint64_t generation = 0;
    int pending = 0;

    std::mutex mu;
    std::condition_variable cv;
    RolloutTask task;
    std::vector<std::thread> workers;
};

struct AlphaZeroPlayer::SearchTreeState {
    std::unique_ptr<Node> root;
};

AlphaZeroPlayer::AlphaZeroPlayer(int c_puct, int n_playout, int seed, int num_threads)
    : c_puct_(c_puct), n_playout_(n_playout), num_threads_(num_threads), player_(1), rng_(seed) {
    if (n_playout_ <= 0) {
        throw std::runtime_error("n_playout must be > 0");
    }
    tree_state_ = std::make_unique<SearchTreeState>();
}

AlphaZeroPlayer::~AlphaZeroPlayer() {
    shutdown_thread_pool();
}

void AlphaZeroPlayer::set_player_ind(int p) {
    player_ = p;
}

void AlphaZeroPlayer::set_n_playout(int n_playout) {
    std::lock_guard<std::mutex> guard(action_mutex_);
    if (n_playout <= 0) {
        throw std::runtime_error("n_playout must be > 0");
    }
    n_playout_ = n_playout;
}

void AlphaZeroPlayer::set_num_threads(int num_threads) {
    std::lock_guard<std::mutex> guard(action_mutex_);
    num_threads_ = num_threads;
    if (num_threads_ <= 1) {
        shutdown_thread_pool();
    }
}

void AlphaZeroPlayer::reset_player() {
    std::lock_guard<std::mutex> guard(action_mutex_);
    if (!tree_state_) {
        tree_state_ = std::make_unique<SearchTreeState>();
    }
    tree_state_->root.reset();
}

void AlphaZeroPlayer::update_with_move(int last_move) {
    std::lock_guard<std::mutex> guard(action_mutex_);
    if (!tree_state_) {
        tree_state_ = std::make_unique<SearchTreeState>();
    }
    if (last_move < 0 || !tree_state_->root) {
        tree_state_->root.reset();
        return;
    }

    Node* root = tree_state_->root.get();
    if (!root->expanded.load(std::memory_order_acquire)) {
        tree_state_->root.reset();
        return;
    }

    std::unique_ptr<Node> next_root;
    for (auto& edge : root->children) {
        if (edge.move == last_move) {
            next_root = std::move(edge.child);
            break;
        }
    }
    tree_state_->root = std::move(next_root);
}

int AlphaZeroPlayer::resolve_thread_count() const {
    int threads = num_threads_;
    if (threads <= 0) {
        const unsigned int hc = std::thread::hardware_concurrency();
        threads = hc == 0U ? 4 : static_cast<int>(hc);
    }
    return std::max(1, threads);
}

void AlphaZeroPlayer::ensure_thread_pool(int threads) {
    if (threads <= 1) {
        return;
    }
    if (pool_state_ && pool_state_->worker_count == threads) {
        return;
    }

    shutdown_thread_pool();
    pool_state_ = std::make_unique<ThreadPoolState>();
    pool_state_->worker_count = threads;
    pool_state_->workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        pool_state_->workers.emplace_back([this, t]() { worker_loop(t); });
    }
}

void AlphaZeroPlayer::shutdown_thread_pool() {
    if (!pool_state_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pool_state_->mu);
        pool_state_->stop = true;
        pool_state_->generation += 1;
    }
    pool_state_->cv.notify_all();
    for (auto& th : pool_state_->workers) {
        if (th.joinable()) {
            th.join();
        }
    }
    pool_state_.reset();
}

void AlphaZeroPlayer::worker_loop(int thread_id) {
    uint64_t observed_generation = 0;

    while (true) {
        ThreadPoolState* state = pool_state_.get();
        if (state == nullptr) {
            return;
        }

        RolloutTask* task = nullptr;
        {
            std::unique_lock<std::mutex> lock(state->mu);
            state->cv.wait(lock, [&]() { return state->stop || state->generation != observed_generation; });
            if (state->stop) {
                return;
            }
            observed_generation = state->generation;
            task = &state->task;
        }

        std::mt19937 thread_rng(static_cast<uint32_t>(task->thread_seeds[static_cast<size_t>(thread_id)] & 0xffffffffULL));
        while (true) {
            const int start = task->next_sim.fetch_add(task->chunk_size);
            if (start >= task->total_playouts) {
                break;
            }
            const int end = std::min(task->total_playouts, start + task->chunk_size);
            for (int sim = start; sim < end; ++sim) {
                (void)sim;
                simulate_one(*task->root, task->root_node, task->c_puct, thread_rng);
            }
        }

        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->pending -= 1;
            if (state->pending == 0) {
                state->cv.notify_all();
            }
        }
    }
}

int AlphaZeroPlayer::get_action(const Board& board) {
    std::lock_guard<std::mutex> guard(action_mutex_);

    const auto& moves = board.availables();
    if (moves.empty()) {
        return -1;
    }

    const int threads = std::min(resolve_thread_count(), n_playout_);
    const int total_playouts = n_playout_;
    const RootSnapshot root = build_root_snapshot(board);

    if (!tree_state_) {
        tree_state_ = std::make_unique<SearchTreeState>();
    }
    if (!tree_state_->root) {
        tree_state_->root = std::make_unique<Node>(1.0F);
    }
    Node* root_node = tree_state_->root.get();

    if (threads == 1) {
        std::mt19937 thread_rng(rng_());
        for (int sim = 0; sim < total_playouts; ++sim) {
            simulate_one(root, root_node, static_cast<float>(c_puct_), thread_rng);
        }
    } else {
        ensure_thread_pool(threads);

        // Generate per-thread seeds on the caller thread to avoid shared RNG races.
        const uint64_t seed_base = static_cast<uint64_t>(rng_());
        std::vector<uint64_t> thread_seeds;
        thread_seeds.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            thread_seeds.push_back(splitmix64(seed_base + static_cast<uint64_t>(t + 1)));
        }

        if (!pool_state_) {
            throw std::runtime_error("thread pool is not initialized");
        }

        auto root_ptr = std::make_shared<RootSnapshot>(root);

        {
            std::unique_lock<std::mutex> lock(pool_state_->mu);
            pool_state_->task.root = root_ptr;
            pool_state_->task.root_node = root_node;
            pool_state_->task.thread_seeds = std::move(thread_seeds);
            pool_state_->task.total_playouts = total_playouts;
            pool_state_->task.c_puct = static_cast<float>(c_puct_);
            pool_state_->task.chunk_size = std::max(16, total_playouts / (threads * 32));
            pool_state_->task.next_sim.store(0);

            pool_state_->pending = threads;
            pool_state_->generation += 1;
            pool_state_->cv.notify_all();

            pool_state_->cv.wait(lock, [&]() { return pool_state_->pending == 0; });
            pool_state_->task.root_node = nullptr;
        }
    }

    if (!root_node->expanded.load(std::memory_order_acquire)) {
        return moves[0];
    }

    int best_move = moves[0];
    int best_visits = -1;
    double best_q = -std::numeric_limits<double>::infinity();
    for (const auto& edge : root_node->children) {
        if (edge.move < 0 || edge.move >= root.width * root.height) {
            continue;
        }
        if (root.pos[static_cast<size_t>(edge.move)] < 0) {
            continue;
        }
        const int cv = edge.child->visits.load(std::memory_order_relaxed);
        const int cs = edge.child->value_sum.load(std::memory_order_relaxed);
        const double q = (cv > 0) ? (static_cast<double>(cs) / static_cast<double>(cv)) : -1e9;
        if (cv > best_visits || (cv == best_visits && q > best_q)) {
            best_visits = cv;
            best_q = q;
            best_move = edge.move;
        }
    }

    return best_move;
}

std::vector<float> AlphaZeroPlayer::get_move_probs(const Board& board, float temp) {
    std::lock_guard<std::mutex> guard(action_mutex_);

    const auto& moves = board.availables();
    const int board_size = board.width() * board.height();
    std::vector<float> full_probs(static_cast<size_t>(board_size), 0.0F);
    if (moves.empty()) {
        return full_probs;
    }

    const int threads = std::min(resolve_thread_count(), n_playout_);
    const int total_playouts = n_playout_;
    const RootSnapshot root = build_root_snapshot(board);

    if (!tree_state_) {
        tree_state_ = std::make_unique<SearchTreeState>();
    }
    if (!tree_state_->root) {
        tree_state_->root = std::make_unique<Node>(1.0F);
    }
    Node* root_node = tree_state_->root.get();

    if (threads == 1) {
        std::mt19937 thread_rng(rng_());
        for (int sim = 0; sim < total_playouts; ++sim) {
            simulate_one(root, root_node, static_cast<float>(c_puct_), thread_rng);
        }
    } else {
        ensure_thread_pool(threads);

        const uint64_t seed_base = static_cast<uint64_t>(rng_());
        std::vector<uint64_t> thread_seeds;
        thread_seeds.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            thread_seeds.push_back(splitmix64(seed_base + static_cast<uint64_t>(t + 1)));
        }

        if (!pool_state_) {
            throw std::runtime_error("thread pool is not initialized");
        }

        auto root_ptr = std::make_shared<RootSnapshot>(root);
        {
            std::unique_lock<std::mutex> lock(pool_state_->mu);
            pool_state_->task.root = root_ptr;
            pool_state_->task.root_node = root_node;
            pool_state_->task.thread_seeds = std::move(thread_seeds);
            pool_state_->task.total_playouts = total_playouts;
            pool_state_->task.c_puct = static_cast<float>(c_puct_);
            pool_state_->task.chunk_size = std::max(16, total_playouts / (threads * 32));
            pool_state_->task.next_sim.store(0);

            pool_state_->pending = threads;
            pool_state_->generation += 1;
            pool_state_->cv.notify_all();

            pool_state_->cv.wait(lock, [&]() { return pool_state_->pending == 0; });
            pool_state_->task.root_node = nullptr;
        }
    }

    if (!root_node->expanded.load(std::memory_order_acquire)) {
        const float uniform = 1.0F / static_cast<float>(moves.size());
        for (int mv : moves) {
            full_probs[static_cast<size_t>(mv)] = uniform;
        }
        return full_probs;
    }

    std::vector<int> legal_moves;
    std::vector<double> logits;
    legal_moves.reserve(root_node->children.size());
    logits.reserve(root_node->children.size());

    const float safe_temp = std::max(1e-6F, temp);
    int best_move = moves[0];
    int best_visits = -1;
    for (const auto& edge : root_node->children) {
        if (edge.move < 0 || edge.move >= board_size) {
            continue;
        }
        if (root.pos[static_cast<size_t>(edge.move)] < 0) {
            continue;
        }
        const int cv = edge.child->visits.load(std::memory_order_relaxed);
        if (cv <= 0) {
            continue;
        }
        legal_moves.push_back(edge.move);
        logits.push_back(std::log(static_cast<double>(cv) + 1e-10) / static_cast<double>(safe_temp));
        if (cv > best_visits) {
            best_visits = cv;
            best_move = edge.move;
        }
    }

    if (legal_moves.empty()) {
        full_probs[static_cast<size_t>(best_move)] = 1.0F;
        return full_probs;
    }

    const double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    for (double& v : logits) {
        v = std::exp(v - max_logit);
        sum_exp += v;
    }

    if (sum_exp <= 0.0) {
        const float uniform = 1.0F / static_cast<float>(legal_moves.size());
        for (int mv : legal_moves) {
            full_probs[static_cast<size_t>(mv)] = uniform;
        }
        return full_probs;
    }

    for (size_t i = 0; i < legal_moves.size(); ++i) {
        full_probs[static_cast<size_t>(legal_moves[i])] = static_cast<float>(logits[i] / sum_exp);
    }
    return full_probs;
}
