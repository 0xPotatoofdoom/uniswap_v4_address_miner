import numpy as np
import cupy as cp
from numba import cuda
import eth_utils
from eth_hash.auto import keccak
import time
from datetime import datetime
import os
import threading
import queue
import json
from pathlib import Path
import signal
import sys

# Challenge constants
DEPLOYER_ADDRESS = "0x6a6EA46de100E2416854968C68A48C44213a2A06"
INIT_CODE_HASH = bytes.fromhex("e6d6849bca7ac3ebf3931038d5a8fbfc15a36f1e2933f7a77abfddf0b62c6527")
MINIMUM_SCORE = 127

# Convert constants to GPU-friendly format
DEPLOYER_BYTES = bytes.fromhex(DEPLOYER_ADDRESS[2:])
PREFIX_BYTES = bytes([0xff])

# Constants from original implementation
ROTC = cuda.to_device(np.array([1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44], dtype=np.int32))
PILN = cuda.to_device(np.array([10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1], dtype=np.int32))
RNDC = cuda.to_device(np.array([
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
], dtype=np.uint64))

# Global state for managing mining operations
class MiningState:
    def __init__(self):
        self.running = True
        self.result_queue = queue.Queue()
        self.total_checked = 0
        self.best_score = 0
        self.checkpoint_file = Path("mining_checkpoint.json")
        self.results_file = Path("good_salts.txt")
        self.stats_lock = threading.Lock()
        
        # Load checkpoint if exists
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                checkpoint = json.load(f)
                self.total_checked = checkpoint.get("total_checked", 0)
                self.best_score = checkpoint.get("best_score", 0)

state = MiningState()

# Original GPU kernel functions (unchanged)
@cuda.jit(device=True)
def generate_random_salt(seed, thread_id, out):
    """Generate a random salt using thread ID and seed"""
    state = seed ^ thread_id
    for i in range(12):
        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5
        out[i] = state & 0xFF

@cuda.jit(device=True)
def keccak256_gpu(input_data, input_len, output):
    """Simple Keccak-256 implementation for GPU"""
    for i in range(32):
        output[i] = 0
        
    for i in range(input_len):
        output[i % 32] ^= input_data[i]
        
        if i % 8 == 0:
            for j in range(31):
                output[j] ^= output[(j + 1) % 32]
            
            temp = output[0]
            for j in range(31):
                output[j] = output[j + 1]
            output[31] = temp

@cuda.jit
def mine_addresses_kernel(seed, addresses, scores):
    """GPU kernel for mining addresses (same as original)"""
    thread_id = cuda.grid(1)
    if thread_id >= addresses.shape[0]:
        return
        
    buffer = cuda.local.array(85, dtype=np.uint8)
    buffer[0] = 0xff
    
    for i in range(20):
        buffer[i + 1] = DEPLOYER_BYTES[i]
    
    for i in range(20):
        buffer[i + 21] = 0
    
    salt = cuda.local.array(12, dtype=np.uint8)
    generate_random_salt(seed, thread_id, salt)
    for i in range(12):
        buffer[i + 41] = salt[i]
        
    for i in range(32):
        buffer[i + 53] = INIT_CODE_HASH[i]
    
    result = cuda.local.array(32, dtype=np.uint8)
    keccak256_gpu(buffer, 85, result)
    
    for i in range(20):
        addresses[thread_id, i] = result[i + 12]
    
    score = 0
    found_four = False
    consecutive_fours = 0
    max_consecutive_fours = 0
    
    leading_zeros = 0
    for i in range(20):
        high_nibble = addresses[thread_id, i] >> 4
        low_nibble = addresses[thread_id, i] & 0x0F
        
        if not found_four:
            if high_nibble == 0:
                score += 10
                leading_zeros += 1
            elif high_nibble == 4:
                found_four = True
                score += 1
            else:
                scores[thread_id] = 0
                return
                
            if not found_four:
                if low_nibble == 0:
                    score += 10
                    leading_zeros += 1
                elif low_nibble == 4:
                    found_four = True
                    score += 1
                else:
                    scores[thread_id] = 0
                    return
        
        if high_nibble == 4:
            consecutive_fours += 1
            score += 1
        else:
            max_consecutive_fours = max(max_consecutive_fours, consecutive_fours)
            consecutive_fours = 0
            
        if low_nibble == 4:
            consecutive_fours += 1
            score += 1
        else:
            max_consecutive_fours = max(max_consecutive_fours, consecutive_fours)
            consecutive_fours = 0
    
    max_consecutive_fours = max(max_consecutive_fours, consecutive_fours)
    
    if max_consecutive_fours >= 4:
        score += 40
        
        for i in range(19):
            high_nibble = addresses[thread_id, i] >> 4
            low_nibble = addresses[thread_id, i] & 0x0F
            if high_nibble == 4 and consecutive_fours >= 4:
                if i < 19 and (addresses[thread_id, i+1] >> 4) != 4:
                    score += 20
                    break
            if low_nibble == 4 and consecutive_fours >= 4:
                if (addresses[thread_id, i] & 0x0F) != 4:
                    score += 20
                    break
    
    end_fours = 0
    for i in range(19, 17, -1):
        if (addresses[thread_id, i] >> 4) == 4:
            end_fours += 1
        if (addresses[thread_id, i] & 0x0F) == 4:
            end_fours += 1
    if end_fours >= 4:
        score += 20
    
    scores[thread_id] = score

class GPUMiner:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.device = cp.cuda.Device(gpu_id)
        self.context = None
        self.addresses = None
        self.scores = None
        self.threads_per_block = 1024
        self.blocks = 8192
        self.total_threads = self.threads_per_block * self.blocks
        
    def initialize(self):
        self.context = self.device.use()
        self.addresses = cuda.device_array((self.total_threads, 20), dtype=np.uint8)
        self.scores = cuda.device_array(self.total_threads, dtype=np.int32)
        return self
        
    def cleanup(self):
        if self.context:
            self.context.pop()
            
    def mine(self, seed):
        mine_addresses_kernel[self.blocks, self.threads_per_block](seed, self.addresses, self.scores)
        return self.process_results(seed)
        
    def process_results(self, seed):
        cpu_scores = self.scores.copy_to_host()
        max_score_idx = np.argmax(cpu_scores)
        current_best_score = cpu_scores[max_score_idx]
        
        if current_best_score >= MINIMUM_SCORE:
            winning_address = self.addresses[max_score_idx].copy_to_host()
            
            winning_salt = bytearray(32)
            for i in range(20):
                winning_salt[i] = 0
            temp_salt = cuda.local.array(12, dtype=np.uint8)
            generate_random_salt(seed, max_score_idx, temp_salt)
            for i in range(12):
                winning_salt[20 + i] = temp_salt[i]
            
            return {
                'score': current_best_score,
                'address': winning_address,
                'salt': winning_salt
            }
        
        return None

def save_checkpoint():
    """Save current mining progress"""
    with state.stats_lock:
        checkpoint_data = {
            "total_checked": state.total_checked,
            "best_score": state.best_score,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(state.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

def save_result(score: int, salt: str, address: str):
    """Save good results to a file"""
    with open(state.results_file, 'a') as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Score: {score}\nAddress: {address}\nSalt: 0x{salt}\n\n")

def mining_worker(gpu_id):
    """Worker function for each GPU"""
    miner = None
    try:
        miner = GPUMiner(gpu_id).initialize()
        print(f"GPU {gpu_id} initialized successfully")
        
        seed = int.from_bytes(os.urandom(4), 'big') ^ (gpu_id << 24)
        
        while state.running:
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            
            result = miner.mine(seed)
            
            with state.stats_lock:
                state.total_checked += miner.total_threads
            
            if result:
                state.result_queue.put((gpu_id, result))
            
    except Exception as e:
        print(f"Error on GPU {gpu_id}: {e}")
    finally:
        if miner:
            miner.cleanup()

def process_results():
    """Process results from all GPUs"""
    try:
        while state.running:
            try:
                gpu_id, result = state.result_queue.get(timeout=1.0)
                
                if result['score'] > state.best_score:
                    with state.stats_lock:
                        state.best_score = result['score']
                    
                    address_hex = '0x' + ''.join([f'{x:02x}' for x in result['address']])
                    checksum_address = eth_utils.to_checksum_address(address_hex)
                    
                    print(f"\nNew best found on GPU {gpu_id}!")
                    print(f"Score: {result['score']}")
                    print(f"Address: {checksum_address}")
                    print(f"Salt: 0x{result['salt'].hex()}")
                    
                    save_result(result['score'], result['salt'].hex(), checksum_address)
                    save_checkpoint()
            
            except queue.Empty:
                continue
                
    except Exception as e:
        print(f"Error in result processor: {e}")
        state.running = False

def print_stats(start_time):
    """Print mining statistics periodically"""
    last_check = state.total_checked
    last_time = start_time
    
    try:
        while state.running:
            time.sleep(5.0)
            
            current_time = time.time()
            current_checked = state.total_checked
            
            elapsed = current_time - last_time
            checked_diff = current_checked - last_check
            
            rate = checked_diff / elapsed
            total_rate = current_checked / (current_time - start_time)
            
            print(f"\rChecked {current_checked:,} addresses "
                  f"(Current: {rate:.0f}/s, Avg: {total_rate:.0f}/s). "
                  f"Best score: {state.best_score}", end="", flush=True)
            
            last_check = current_checked
            last_time = current_time
            
            save_checkpoint()
            
    except Exception as e:
        print(f"Error in stats printer: {e}")
        state.running = False

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\nShutting down miners...")
    state.running = False

def main():
    """Main function to coordinate mining operations"""
    print("Uniswap v4 CREATE2 Address Mining Challenge - Multi-GPU Edition")
    print("Competition period: Nov 10, 2024 - Dec 1, 2024")
    print(f"Deployer address: {DEPLOYER_ADDRESS}")
    print(f"Init code hash: 0x{INIT_CODE_HASH.hex()}")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Get number of available GPUs
        num_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"\nFound {num_gpus} GPUs")
        
        if num_gpus == 0:
            print("No GPUs available!")
            return
        
        # Start mining threads
        start_time = time.time()

        # Create and start mining threads
        mining_threads = []
        for gpu_id in range(num_gpus):
            thread = threading.Thread(
                target=mining_worker,
                args=(gpu_id,),
                name=f"Miner-GPU{gpu_id}"
            )
            thread.daemon = True
            thread.start()
            mining_threads.append(thread)
            
        # Start result processor thread
        result_thread = threading.Thread(
            target=process_results,
            name="ResultProcessor"
        )
        result_thread.daemon = True
        result_thread.start()
        
        # Start statistics printer thread
        stats_thread = threading.Thread(
            target=print_stats,
            args=(start_time,),
            name="StatsPrinter"
        )
        stats_thread.daemon = True
        stats_thread.start()
        
        # Wait for all threads to complete
        try:
            while state.running:
                if not any(t.is_alive() for t in mining_threads):
                    print("\nAll mining threads have stopped!")
                    break
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        
        finally:
            state.running = False
            
            # Wait for threads to finish
            for thread in mining_threads:
                thread.join(timeout=2.0)
            result_thread.join(timeout=2.0)
            stats_thread.join(timeout=2.0)
            
            # Save final checkpoint
            save_checkpoint()
            
            # Print final statistics
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\n\nFinal Statistics:")
            print(f"Total addresses checked: {state.total_checked:,}")
            print(f"Best score achieved: {state.best_score}")
            print(f"Average rate: {state.total_checked/total_time:,.0f} addresses/second")
            print(f"Total runtime: {total_time:.1f} seconds")
            
    except Exception as e:
        print(f"Error in main thread: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nMining completed")

if __name__ == "__main__":
    main()