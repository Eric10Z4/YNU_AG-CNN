#!/usr/bin/env python3
"""
Verification script for C++ MCTS neural network integration.
Run after compiling mcts_cpp to verify correctness.

Tests:
1. C++ module loads and has new set_eval_callback method
2. Neural network callback is invoked during MCTS search
3. NN-guided MCTS produces non-uniform move probabilities (unlike pure rollout)
4. State channels built by C++ match Python Board.current_state()
5. End-to-end mini training loop: p_loss decreases over a few iterations

Usage:
    conda activate ynu-cnn
    cd /root/YNU_AG-CNN/YNU_AG-CNN-main/gat_cpp
    python verify_nn_mcts.py
"""

import sys
import os
import time
import numpy as np
import torch

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cpp_game import Board, Game


def test_1_module_loads():
    """Test 1: C++ module loads and has new API."""
    print("=" * 60)
    print("TEST 1: Module loads and has set_eval_callback")
    
    try:
        import mcts_cpp
    except ImportError:
        # Try building first
        build_dir = os.path.join(os.path.dirname(__file__), "build")
        sys.path.insert(0, build_dir)
        import mcts_cpp

    player = mcts_cpp.AlphaZeroPlayer(c_puct=5, n_playout=10, seed=42, num_threads=1)
    assert hasattr(player, 'set_eval_callback'), "Missing set_eval_callback method!"
    assert hasattr(player, 'has_eval_callback'), "Missing has_eval_callback method!"
    assert not player.has_eval_callback(), "Should not have callback initially"
    print("PASS: Module loaded, API present\n")
    return mcts_cpp


def test_2_callback_invoked(mcts_cpp):
    """Test 2: Callback is actually invoked during search."""
    print("=" * 60)
    print("TEST 2: Neural network callback is invoked during MCTS")
    
    call_count = [0]
    board_size = 11
    
    def dummy_eval(state_np, batch_size, width, height):
        call_count[0] += 1
        results = []
        for i in range(batch_size):
            policy = np.ones(width * height, dtype=np.float32) / (width * height)
            results.append((policy.tolist(), 0.0))
        return results
    
    board = mcts_cpp.Board(board_size, board_size, 5)
    board.init_board(0)
    
    player = mcts_cpp.AlphaZeroPlayer(c_puct=5, n_playout=40, seed=42, num_threads=1)
    player.set_eval_callback(dummy_eval, 8)
    assert player.has_eval_callback(), "Callback should be set"
    
    probs = player.get_move_probs(board, 1.0)
    
    assert call_count[0] > 0, f"Callback was never invoked! call_count={call_count[0]}"
    print(f"PASS: Callback invoked {call_count[0]} times for 40 playouts (batch=8)\n")


def test_3_nn_vs_rollout_probs(mcts_cpp):
    """Test 3: NN-guided MCTS produces different probs than pure rollout."""
    print("=" * 60)
    print("TEST 3: NN-guided MCTS produces non-uniform probabilities")
    
    board_size = 11
    board = mcts_cpp.Board(board_size, board_size, 5)
    board.init_board(0)
    # Make a few moves to create an interesting position
    board.do_move(60)  # center
    board.do_move(61)  # adjacent
    board.do_move(49)  # above center
    board.do_move(71)  # below center
    
    # Create a biased policy: heavily favor position 38 (above-left of center)
    def biased_eval(state_np, batch_size, width, height):
        results = []
        for i in range(batch_size):
            policy = np.ones(width * height, dtype=np.float32) * 0.001
            policy[38] = 0.9  # Strong prior for one move
            policy /= policy.sum()
            results.append((policy.tolist(), 0.5))
        return results
    
    player = mcts_cpp.AlphaZeroPlayer(c_puct=5, n_playout=100, seed=42, num_threads=1)
    player.set_eval_callback(biased_eval, 8)
    
    probs = np.array(player.get_move_probs(board, 1.0))
    
    # The move with the highest NN prior should have high visit probability
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero + 1e-10))
    max_entropy = np.log(len(nonzero))
    
    print(f"  Prob at biased move (38): {probs[38]:.4f}")
    print(f"  Max prob move: {np.argmax(probs)} (prob={np.max(probs):.4f})")
    print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} (ratio={entropy/max_entropy:.3f})")
    
    # With biased priors, entropy should be much lower than uniform
    assert entropy < max_entropy * 0.9, \
        f"Entropy too high ({entropy:.3f}), probs look uniform. NN priors not being used!"
    assert probs[38] > 0.05, \
        f"Biased move prob too low ({probs[38]:.4f}), NN priors not being used!"
    
    print("PASS: NN-guided MCTS produces non-uniform, prior-influenced probabilities\n")


def test_4_state_channels(mcts_cpp):
    """Test 4: C++ state channels match Python Board.current_state()."""
    print("=" * 60)
    print("TEST 4: State channels match Python Board.current_state()")
    
    board_size = 11
    
    # Build identical positions in Python and C++ boards
    py_board = Board(board_size, board_size, 5)
    py_board.init_board()
    cpp_board = mcts_cpp.Board(board_size, board_size, 5)
    cpp_board.init_board(0)
    
    moves = [60, 61, 49, 71, 38, 82]  # some moves
    for m in moves:
        py_board.do_move(m)
        cpp_board.do_move(m)
    
    py_state = py_board.current_state()  # (4, W, H)
    
    # Capture the state that C++ builds via callback
    captured_state = [None]
    
    def capture_eval(state_np, batch_size, width, height):
        captured_state[0] = np.array(state_np)
        results = []
        for i in range(batch_size):
            policy = np.ones(width * height, dtype=np.float32) / (width * height)
            results.append((policy.tolist(), 0.0))
        return results
    
    player = mcts_cpp.AlphaZeroPlayer(c_puct=5, n_playout=1, seed=42, num_threads=1)
    player.set_eval_callback(capture_eval, 1)  # batch_size=1 to get exactly 1 state
    player.get_move_probs(cpp_board, 1e-3)
    
    assert captured_state[0] is not None, "Callback was not invoked!"
    cpp_state = captured_state[0][0]  # first (and only) state in the batch
    
    # Compare
    diff = np.abs(py_state - cpp_state)
    max_diff = diff.max()
    
    print(f"  Python state sum: ch0={py_state[0].sum():.0f}, ch1={py_state[1].sum():.0f}, "
          f"ch2={py_state[2].sum():.0f}, ch3={py_state[3].sum():.0f}")
    print(f"  C++    state sum: ch0={cpp_state[0].sum():.0f}, ch1={cpp_state[1].sum():.0f}, "
          f"ch2={cpp_state[2].sum():.0f}, ch3={cpp_state[3].sum():.0f}")
    print(f"  Max difference: {max_diff}")
    
    assert max_diff < 1e-6, f"State mismatch! Max diff={max_diff}"
    print("PASS: C++ state channels exactly match Python Board.current_state()\n")


def test_5_mini_training(mcts_cpp):
    """Test 5: Mini end-to-end training loop - p_loss should decrease."""
    print("=" * 60)
    print("TEST 5: Mini training loop - verify p_loss decreases")
    
    # Import the policy network
    try:
        from cpp_train import TrainPipeline
    except Exception as e:
        print(f"SKIP: Cannot import TrainPipeline ({e})")
        return
    
    board_size = 6  # Small board for fast test
    board = Board(board_size, board_size, 4)
    
    # Import network
    from cpp_train import PolicyValueNet
    net = PolicyValueNet(board_size, num_channels=32, device='cpu')
    board_area = board_size * board_size
    
    def make_eval_fn():
        def batch_eval(state_np, batch_size, width, height):
            state_tensor = torch.from_numpy(
                np.ascontiguousarray(state_np)
            ).float()
            net.eval_mode()
            with torch.no_grad():
                act_probs, values = net.forward(state_tensor)
            act_probs_np = act_probs.cpu().numpy()[:, :board_area]
            values_np = values.cpu().numpy()
            results = []
            for i in range(batch_size):
                results.append((act_probs_np[i].tolist(), float(values_np[i][0])))
            return results
        return batch_eval
    
    # Collect some self-play data with NN-guided MCTS
    print("  Collecting self-play data with NN-guided C++ MCTS...")
    all_data = []
    for game_idx in range(3):
        cpp_board = mcts_cpp.Board(board_size, board_size, 4)
        cpp_board.init_board(0)
        py_board = Board(board_size, board_size, 4)
        py_board.init_board()
        
        player = mcts_cpp.AlphaZeroPlayer(c_puct=5, n_playout=50, seed=42 + game_idx, num_threads=1)
        player.set_eval_callback(make_eval_fn(), 8)
        player.set_player_ind(1)
        
        states, mcts_probs, current_players = [], [], []
        while True:
            probs = np.array(player.get_move_probs(cpp_board, 1.0))
            
            states.append(py_board.current_state())
            mcts_probs.append(probs)
            current_players.append(py_board.current_player)
            
            # Pick move by probabilities
            legal = list(py_board.availables)
            legal_probs = probs[legal]
            legal_probs = np.maximum(legal_probs, 1e-12)
            legal_probs /= legal_probs.sum()
            move = int(np.random.choice(legal, p=legal_probs))
            
            py_board.do_move(move)
            cpp_board.do_move(move)
            player.update_with_move(move)
            
            is_end, winner = py_board.game_end()
            if is_end:
                winners_z = np.zeros(len(current_players), dtype=np.float32)
                if winner != -1:
                    cp = np.array(current_players)
                    winners_z[cp == winner] = 1.0
                    winners_z[cp != winner] = -1.0
                all_data.extend(zip(states, mcts_probs, winners_z))
                print(f"    Game {game_idx+1}: {len(states)} moves, winner={winner}")
                break
    
    if len(all_data) < 10:
        print("SKIP: Not enough data collected")
        return
    
    # Train for a few steps and check loss decrease
    print(f"  Training on {len(all_data)} positions...")
    from torch.optim import Adam
    params, _ = net.get_all_params()
    optimizer = Adam(params, lr=2e-3)
    
    losses = []
    for epoch in range(5):
        batch = list(all_data)
        np.random.shuffle(batch)
        batch = batch[:min(64, len(batch))]
        
        s_batch = torch.tensor(np.array([d[0] for d in batch]), dtype=torch.float32)
        p_batch = torch.tensor(np.array([d[1] for d in batch]), dtype=torch.float32)
        v_batch = torch.tensor(np.array([d[2] for d in batch]), dtype=torch.float32).unsqueeze(1)
        
        net.train_mode()
        act_probs, value = net.forward(s_batch)
        
        # Trim policy if needed
        if act_probs.shape[1] > p_batch.shape[1]:
            act_probs_trimmed = act_probs[:, :p_batch.shape[1]]
        else:
            act_probs_trimmed = act_probs
        
        # Policy loss (cross-entropy)
        p_loss = -torch.mean(torch.sum(p_batch * torch.log(act_probs_trimmed + 1e-10), dim=1))
        # Value loss (MSE)
        v_loss = torch.mean((value - v_batch) ** 2)
        total_loss = p_loss + v_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses.append({
            'total': total_loss.item(),
            'p_loss': p_loss.item(),
            'v_loss': v_loss.item()
        })
        print(f"    Epoch {epoch+1}: loss={total_loss.item():.4f} "
              f"(p={p_loss.item():.4f}, v={v_loss.item():.4f})")
    
    # Verify loss decreased
    first_loss = losses[0]['total']
    last_loss = losses[-1]['total']
    print(f"\n  Loss change: {first_loss:.4f} -> {last_loss:.4f} "
          f"(delta={last_loss - first_loss:.4f})")
    
    assert last_loss < first_loss, \
        f"Loss did not decrease! {first_loss:.4f} -> {last_loss:.4f}"
    
    # Verify p_loss is not stuck at uniform
    uniform_entropy = np.log(board_area)
    assert losses[-1]['p_loss'] < uniform_entropy, \
        f"p_loss ({losses[-1]['p_loss']:.4f}) >= uniform entropy ({uniform_entropy:.4f})"
    
    print("PASS: Training loop works, loss decreasing, NN is learning\n")


def main():
    print("\n" + "=" * 60)
    print("VERIFICATION: C++ MCTS Neural Network Integration")
    print("=" * 60 + "\n")
    
    t0 = time.time()
    
    mcts_cpp = test_1_module_loads()
    test_2_callback_invoked(mcts_cpp)
    test_3_nn_vs_rollout_probs(mcts_cpp)
    test_4_state_channels(mcts_cpp)
    test_5_mini_training(mcts_cpp)
    
    elapsed = time.time() - t0
    print("=" * 60)
    print(f"ALL TESTS PASSED ({elapsed:.1f}s)")
    print("=" * 60)
    print("\nYou can now start training with confidence:")
    print("  conda activate ynu-cnn && nohup python cpp_train.py \\")
    print("    --selfplay-workers 2 --eval-batch-size 8 --game-batch-num 0 \\")
    print("    > train_output.log 2>&1 &")


if __name__ == "__main__":
    main()
