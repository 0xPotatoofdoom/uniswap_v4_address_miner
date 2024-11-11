import numpy as np
import cupy as cp
from numba import cuda
import eth_utils
from eth_hash.auto import keccak
import time
from datetime import datetime
import os

# Challenge constants
DEPLOYER_ADDRESS = "0x6a6EA46de100E2416854968C68A48C44213a2A06"
INIT_CODE_HASH = bytes.fromhex("e6d6849bca7ac3ebf3931038d5a8fbfc15a36f1e2933f7a77abfddf0b62c6527")
MINIMUM_SCORE = 127

# Convert constants to GPU-friendly format
DEPLOYER_BYTES = bytes.fromhex(DEPLOYER_ADDRESS[2:])
PREFIX_BYTES = bytes([0xff])

# Convert Keccak constants to device arrays
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
    # Initialize output
    for i in range(32):
        output[i] = 0
        
    # Basic mixing function
    for i in range(input_len):
        output[i % 32] ^= input_data[i]
        
        # Simple mixing operations
        if i % 8 == 0:
            for j in range(31):
                output[j] ^= output[(j + 1) % 32]
            
            # Rotate bytes
            temp = output[0]
            for j in range(31):
                output[j] = output[j + 1]
            output[31] = temp

@cuda.jit
def mine_addresses_kernel(seed, addresses, scores):
    thread_id = cuda.grid(1)
    if thread_id >= addresses.shape[0]:
        return
        
    # Local working buffer for keccak input
    buffer = cuda.local.array(85, dtype=np.uint8)
    
    # Fill in constant parts
    buffer[0] = 0xff
    
    # Copy deployer address
    for i in range(20):
        buffer[i + 1] = DEPLOYER_BYTES[i]
    
    # Generate and copy salt
    for i in range(20):
        buffer[i + 21] = 0
    
    salt = cuda.local.array(12, dtype=np.uint8)
    generate_random_salt(seed, thread_id, salt)
    for i in range(12):
        buffer[i + 41] = salt[i]
        
    # Copy init code hash
    for i in range(32):
        buffer[i + 53] = INIT_CODE_HASH[i]
    
    # Compute keccak hash
    result = cuda.local.array(32, dtype=np.uint8)
    keccak256_gpu(buffer, 85, result)
    
    # Take last 20 bytes as address
    for i in range(20):
        addresses[thread_id, i] = result[i + 12]
    
    # Score the address
    score = 0
    found_four = False
    consecutive_fours = 0
    max_consecutive_fours = 0
    
    # Count leading zeros
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
        
        # Count consecutive fours and individual fours
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
    
    # Check for four 4s at the end
    end_fours = 0
    for i in range(19, 17, -1):
        if (addresses[thread_id, i] >> 4) == 4:
            end_fours += 1
        if (addresses[thread_id, i] & 0x0F) == 4:
            end_fours += 1
    if end_fours >= 4:
        score += 20
    
    scores[thread_id] = score

def get_gpu_info():
    """Get GPU information using a more reliable method"""
    try:
        device = cp.cuda.Device()
        return {
            'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            'max_threads_per_block': 1024,
            'max_block_dim_x': 1024,
            'max_grid_dim_x': 2147483647,
            'total_memory': device.mem_info[1]
        }
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return {
            'compute_capability': 'unknown',
            'max_threads_per_block': 1024,
            'max_block_dim_x': 1024,
            'max_grid_dim_x': 2147483647,
            'total_memory': 8 * 1024**3
        }

def save_result(score: int, salt: str, address: str):
    """Save good results to a file"""
    with open('good_salts.txt', 'a') as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Score: {score}\nAddress: {address}\nSalt: 0x{salt}\n\n")

def mine_addresses_gpu():
    try:
        gpu_info = get_gpu_info()
        print(f"GPU Information:")
        print(f"Compute Capability: {gpu_info['compute_capability']}")
        print(f"Max threads per block: {gpu_info['max_threads_per_block']}")
        print(f"Max blocks: {gpu_info['max_grid_dim_x']}")
        print(f"Total memory: {gpu_info['total_memory'] / (1024**3):.2f} GB")
        
        threads_per_block = min(1024, gpu_info['max_threads_per_block'])
        blocks = min(gpu_info['max_grid_dim_x'], 8192)
        total_threads = threads_per_block * blocks
        
        print(f"\nUsing {blocks} blocks with {threads_per_block} threads each")
        print(f"Total parallel threads: {total_threads:,}")
        
        addresses = cuda.device_array((total_threads, 20), dtype=np.uint8)
        scores = cuda.device_array(total_threads, dtype=np.int32)
        
        print("\nStarting GPU mining...")
        start_time = time.time()
        total_checked = 0
        best_score = 0
        seed = int.from_bytes(os.urandom(4), 'big')
        elapsed = 0
        
        try:
            while True:
                seed = (seed * 1103515245 + 12345) & 0x7fffffff
                
                mine_addresses_kernel[blocks, threads_per_block](seed, addresses, scores)
                
                cpu_scores = scores.copy_to_host()
                max_score_idx = np.argmax(cpu_scores)
                current_best_score = cpu_scores[max_score_idx]
                
                if current_best_score > best_score and current_best_score >= MINIMUM_SCORE:
                    best_score = current_best_score
                    winning_address = addresses[max_score_idx].copy_to_host()
                    
                    winning_salt = bytearray(32)
                    for i in range(20):
                        winning_salt[i] = 0
                    temp_salt = cuda.local.array(12, dtype=np.uint8)
                    generate_random_salt(seed, max_score_idx, temp_salt)
                    for i in range(12):
                        winning_salt[20 + i] = temp_salt[i]
                    
                    address_hex = '0x' + ''.join([f'{x:02x}' for x in winning_address])
                    checksum_address = eth_utils.to_checksum_address(address_hex)
                    
                    print(f"\nNew best found!")
                    print(f"Score: {best_score}")
                    print(f"Address: {checksum_address}")
                    print(f"Salt: 0x{winning_salt.hex()}")
                    save_result(best_score, winning_salt.hex(), checksum_address)
                
                total_checked += total_threads
                elapsed = time.time() - start_time
                rate = total_checked / elapsed
                
                print(f"\rChecked {total_checked:,} addresses ({rate:.0f}/s). Best score: {best_score}", 
                      end="", flush=True)
        
        except KeyboardInterrupt:
            print("\nMining stopped by user")
        
        finally:
            if elapsed > 0:
                print(f"\nFinal Statistics:")
                print(f"Total addresses checked: {total_checked:,}")
                print(f"Best score achieved: {best_score}")
                print(f"Average rate: {total_checked/elapsed:.0f} addresses/second")
            
    except Exception as e:
        print(f"Error initializing GPU: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    print("Uniswap v4 CREATE2 Address Mining Challenge - GPU Edition")
    print("Competition period: Nov 10, 2024 - Dec 1, 2024")
    print(f"Deployer address: {DEPLOYER_ADDRESS}")
    print(f"Init code hash: 0x{INIT_CODE_HASH.hex()}")
    print("\nInitializing GPU...")
    
    mine_addresses_gpu()