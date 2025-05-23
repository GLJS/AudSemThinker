from pathlib import Path
from collections import defaultdict

def get_shard_pattern(dir_path: Path, skip_last: bool = False) -> str | None:
    """
    Generate shard pattern for tar files in directory using nested braceexpansion.
    
    Args:
        dir_path (Path): Directory path containing tar files
        skip_last (bool): Whether to exclude the last tar file in the sorted list
        
    Returns:
        str | None: Nested brace expansion pattern for tar files, or None if no tar files found
    """
    # Get all tar files and extract their shard numbers
    shard_numbers = []
    for file in dir_path.glob("*.tar"):
        try:
            shard_num = int(file.stem)  # Remove .tar and convert to int
            shard_numbers.append(shard_num)
        except ValueError:
            continue
    
    if not shard_numbers:
        return None
    
    # Sort numbers and optionally remove the last one
    shard_numbers.sort()
    if skip_last and len(shard_numbers) > 0:
        shard_numbers.pop()
    
    # Group numbers by their digit length
    digit_groups = defaultdict(list)
    for num in shard_numbers:
        digit_groups[len(str(num))].append(num)
    
    # Process each digit group
    patterns = []
    for digit_len, numbers in sorted(digit_groups.items()):
        # Group consecutive numbers within each digit length
        ranges = []
        start = numbers[0]
        prev = start
        
        for curr in numbers[1:] + [None]:
            if curr is None or curr != prev + 1:
                # End of a continuous range
                end = prev
                if start == end:
                    ranges.append(str(start))
                else:
                    # Wrap each range in braces
                    ranges.append(f"{{{start}..{end}}}")
                if curr is not None:
                    start = curr
            prev = curr if curr is not None else prev
        
        # Join ranges for this digit length
        if len(ranges) > 1:
            patterns.append(f"{','.join(ranges)}")
        else:
            patterns.append(ranges[0])
    
    # Join all digit groups
    if len(patterns) > 1:
        final_pattern = f"{{{','.join(patterns)}}}"
    else:
        final_pattern = patterns[0]
    
    return str(dir_path / f"{final_pattern}.tar") 