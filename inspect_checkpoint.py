import argparse
import torch

def format_value(value, depth=0, max_depth=2):
    indent = "  " * depth
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
    elif isinstance(value, dict):
        if depth >= max_depth:
            return f"dict with {len(value)} keys: {list(value.keys())[:5]}..."
        
        # Check if the dictionary is a state_dict or similar huge tensor collection
        has_tensors = any(isinstance(v, torch.Tensor) for v in value.values())
        if len(value) > 20 or has_tensors:
            key_summary = []
            for k, v in list(value.items())[:5]:
                if isinstance(v, torch.Tensor):
                    key_summary.append(f"'{k}': Tensor(shape={list(v.shape)})")
                else:
                    key_summary.append(f"'{k}': {type(v).__name__}")
            suffix = ", ..." if len(value) > 5 else ""
            return f"dict with {len(value)} keys: {{{', '.join(key_summary)}{suffix}}}"
        
        lines = []
        for k, v in value.items():
            lines.append(f"\n{indent}  '{k}': {format_value(v, depth + 1, max_depth)}")
        return "{" + "".join(lines) + "\n" + indent + "}"
    elif isinstance(value, list):
        if len(value) > 10:
            return f"list of length {len(value)}: [{', '.join(str(x) for x in value[:3])}, ...]"
        # format elements inside list if they are dicts or tensors
        formatted_list = [format_value(x, depth + 1, max_depth) for x in value]
        return f"[{', '.join(formatted_list)}]"
    else:
        return repr(value)

def inspect_checkpoint(pth_path):
    print(f"Loading checkpoint from: {pth_path}")
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        print(f"Checkpoint contains a dictionary with {len(checkpoint)} keys:")
        for key, value in checkpoint.items():
            if key == 'state_dict':
                if isinstance(value, dict):
                    print(f"  - '{key}': dict with {len(value)} keys (skipped printing values)")
                else:
                    print(f"  - '{key}': [value type: {type(value)} (skipped printing)]")
            else:
                print(f"  - '{key}': {format_value(value, depth=1)}")
    else:
        print(f"Loaded object is of type: {type(checkpoint)}")

def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch checkpoint (.pth) files.")
    parser.add_argument("pth_path", type=str, help="Path to the .pth file.")
    args = parser.parse_args()
    inspect_checkpoint(args.pth_path)

if __name__ == "__main__":
    main()
