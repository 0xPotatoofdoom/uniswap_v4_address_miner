import eth_utils
from eth_hash.auto import keccak
import time
import os
from multiprocessing import Pool, cpu_count, Value, Lock
import psutil
import collections
from datetime import datetime
import json
from typing import Dict, List, Tuple
import signal
import random  # Added missing import
import atexit

# Challenge constants
DEPLOYER_ADDRESS = "0x6a6EA46de100E2416854968C68A48C44213a2A06"
INIT_CODE_HASH = bytes.fromhex("e6d6849bca7ac3ebf3931038d5a8fbfc15a36f1e2933f7a77abfddf0b62c6527")
MINIMUM_SCORE = 127  # We need to beat 126

# Global pool variable for proper cleanup
pool = None

def cleanup():
    """Cleanup function to properly close the pool"""
    global pool
    if pool:
        pool.close()
        pool.join()

# Register cleanup function
atexit.register(cleanup)

def create_salt(random_part: bytes) -> bytes:
    """Creates a valid salt with first 20 bytes as zeros"""
    return bytes(20) + random_part

def calculate_create2_address(sender_address: str, salt: bytes, init_code_hash: bytes) -> str:
    """Calculate the CREATE2 deployment address."""
    sender_address = sender_address.lower().replace('0x', '')
    prefix = bytes([0xff])
    components = prefix + bytes.fromhex(sender_address) + salt + init_code_hash
    address_bytes = keccak(components)
    raw_address = '0x' + address_bytes[-20:].hex()
    return eth_utils.to_checksum_address(raw_address)

def is_valid_address(address: str) -> bool:
    """Check if address has first nonzero nibble as 4"""
    addr_hex = address[2:].lower()
    for nibble in addr_hex:
        if nibble != '0':
            return nibble == '4'
    return False

def score_address(address: str) -> tuple[int, dict]:
    """Score an address based on the competition criteria."""
    address = address[2:].lower()
    
    if not is_valid_address('0x' + address):
        return 0, {'invalid': 'First nonzero nibble must be 4'}
    
    score = 0
    breakdown = {
        'leading_zeros': 0,
        'four_fours': 0,
        'after_fours': 0,
        'last_fours': 0,
        'individual_fours': 0
    }
    
    # Count leading zeros
    for char in address:
        if char == '0':
            breakdown['leading_zeros'] += 10
            score += 10
        else:
            break
    
    # Check for four 4s in a row
    if '4444' in address:
        first_4444_idx = address.index('4444')
        breakdown['four_fours'] = 40
        score += 40
        
        if len(address) > first_4444_idx + 4:
            next_char = address[first_4444_idx + 4]
            if next_char != '4':
                breakdown['after_fours'] = 20
                score += 20
    
    # Check last 4 nibbles
    if address[-4:] == '4444':
        breakdown['last_fours'] = 20
        score += 20
    
    # Count individual 4s
    four_count = address.count('4')
    breakdown['individual_fours'] = four_count
    score += four_count
    
    return score, breakdown

def generate_smart_random_part() -> bytes:
    """Generate random bytes with higher probability of desired patterns"""
    result = bytearray(12)
    
    # Different strategies for generating the random part
    strategy = random.random()
    
    if strategy < 0.4:  # Pure random
        return os.urandom(12)
    
    elif strategy < 0.7:  # Include some 4s
        for i in range(12):
            if random.random() < 0.3:
                result[i] = 0x44  # Hex for '4'
            else:
                result[i] = random.randint(0, 255)
    
    else:  # Try to create sequences of 4s
        seq_start = random.randint(0, 8)  # Leave room for at least 4 bytes
        for i in range(12):
            if seq_start <= i < seq_start + 4:
                result[i] = 0x44
            else:
                result[i] = random.randint(0, 255)
    
    return bytes(result)

def process_batch(args) -> tuple[int, str, str, dict]:
    """Process a batch of addresses"""
    batch_size, worker_id = args
    best_score = 0
    best_salt = None
    best_address = None
    stats = {'checked': 0, 'valid': 0}
    
    try:
        for _ in range(batch_size):
            random_part = generate_smart_random_part()
            salt = create_salt(random_part)
            address = calculate_create2_address(DEPLOYER_ADDRESS, salt, INIT_CODE_HASH)
            
            stats['checked'] += 1
            
            if is_valid_address(address):
                stats['valid'] += 1
                score, _ = score_address(address)
                
                if score > best_score:
                    best_score = score
                    best_salt = salt.hex()
                    best_address = address
        
        return best_score, best_salt, best_address, stats
    
    except Exception as e:
        print(f"Worker {worker_id} error: {str(e)}")
        return 0, None, None, stats

def save_result(score: int, salt: str, address: str):
    """Save good results to a file"""
    with open('good_salts.txt', 'a') as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Score: {score}\nAddress: {address}\nSalt: 0x{salt}\n\n")

def mine_addresses():
    """Main mining loop"""
    global pool
    print("Starting optimized address mining...")
    print(f"Using {cpu_count()} CPU cores")
    
    total_checked = 0
    best_overall_score = 0
    start_time = time.time()
    batch_size = 10000
    
    try:
        pool = Pool(cpu_count())
        
        while True:
            worker_args = [(batch_size, i) for i in range(cpu_count())]
            results = pool.map(process_batch, worker_args)
            
            for score, salt, address, stats in results:
                total_checked += stats['checked']
                
                if score > best_overall_score and score > MINIMUM_SCORE:
                    best_overall_score = score
                    print(f"\nNew best found!")
                    print(f"Score: {score}")
                    print(f"Address: {address}")
                    print(f"Salt: 0x{salt}")
                    save_result(score, salt, address)
            
            # Print progress
            elapsed = time.time() - start_time
            rate = total_checked / elapsed
            print(f"\rChecked {total_checked:,} addresses ({rate:.0f}/s). Best score: {best_overall_score}", 
                  end="", flush=True)
    
    except KeyboardInterrupt:
        print("\nMining stopped by user")
    
    finally:
        if pool:
            pool.close()
            pool.join()
        
        print(f"\nFinal Statistics:")
        print(f"Total addresses checked: {total_checked:,}")
        print(f"Best score achieved: {best_overall_score}")
        print(f"Average rate: {total_checked/elapsed:.0f} addresses/second")

if __name__ == "__main__":
    print("Uniswap v4 CREATE2 Address Mining Challenge")
    print("Competition period: Nov 10, 2024 - Dec 1, 2024")
    print(f"Deployer address: {DEPLOYER_ADDRESS}")
    print(f"Init code hash: 0x{INIT_CODE_HASH.hex()}")
    
    try:
        mine_addresses()
    finally:
        cleanup()